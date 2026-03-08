import time
import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from collections import namedtuple
from tqdm import tqdm, trange

import wandb

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Creating a class to store the results
TrainingResults = namedtuple('TrainingResults', ['train_loss', 'val_loss', 'test_loss', 'train_acc', 'val_acc', 'test_acc'])


class Trainer:
    def __init__(self, model, loss_fn, optimizer, use_wandb=False, wandb_run=None, device="cpu", is_master=True):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.is_master = is_master

        # W&B integration
        self.use_wandb = use_wandb
        self.wandb_run = wandb_run

        # Only master rank watches the model
        if self.is_master and self.use_wandb and self.wandb_run is not None:
            self.wandb_run.watch(self.model, log="all", log_freq=100)

    def train_batch(self, batch):
        raise NotImplementedError()

    def test_batch(self, batch):
        raise NotImplementedError()

    def fit(self, dl_train, dl_val, dl_test, config, checkpoints=None):
        update_lr_epoch_num = config["training"]["update_lr_epoch_num"]
        lr_factor = config["training"]["lr_factor"]
        num_epochs = config["training"]["num_epochs"]
        early_stopping = config["training"]["early_stopping"]
        print_every = config["training"]["print_every"]
        max_batches_per_epoch = config["training"]["max_batches_per_epoch"]

        best_acc = None
        epochs_without_improvement = 0
        checkpoint_path = f"{checkpoints}.pt" if checkpoints else None

        train_loss, train_acc = [], []
        val_loss, val_acc = [], []
        test_loss, test_acc = [], []

        # Load checkpoint if it exists
        if checkpoint_path and os.path.isfile(checkpoint_path):
            if self.is_master:
                print(f"*** Loading checkpoint file from {checkpoint_path}")

            # Map location ensures Rank 3 loads into GPU 3, not GPU 0
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # If the model is already wrapped in DDP, use .module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(checkpoint["model_state"])
            else:
                self.model.load_state_dict(checkpoint["model_state"])

            if "optimizer_state" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])

            best_acc = checkpoint.get("best_acc", best_acc)
            epochs_without_improvement = checkpoint.get("ewi", epochs_without_improvement)

        if self.is_master:
            print("Start training")

        for epoch in range(num_epochs):
            # Essential for correct shuffling in DDP
            if hasattr(dl_train.sampler, 'set_epoch'):
                dl_train.sampler.set_epoch(epoch)

            self.model.train()
            train_result = self._run_epoch(dl_train, train=True, max_batches=max_batches_per_epoch)
            train_loss.extend(train_result["loss"])
            train_acc.append(train_result["accuracy"])

            self.model.eval()
            val_result = self._run_epoch(dl_val, train=False, max_batches=max_batches_per_epoch)
            val_loss.extend(val_result["loss"])
            val_acc.append(val_result["accuracy"])

            test_result = self._run_epoch(dl_test, train=False, max_batches=max_batches_per_epoch)
            test_loss.extend(test_result["loss"])
            test_acc.append(test_result["accuracy"])

            # ONLY Master prints and logs
            if self.is_master:
                if epoch % print_every == 0 or epoch == num_epochs - 1:
                    print(f"--- EPOCH {epoch + 1}/{num_epochs} ---")
                    print(f"  Train Loss: {train_result['loss'][0]:.4f} | Train Acc: {train_result['accuracy']:.2f}%")
                    print(f"  Val Loss: {val_result['loss'][0]:.4f} | Val Acc: {val_result['accuracy']:.2f}%")
                    print(f"  Test Loss: {test_result['loss'][0]:.4f} | Test Acc: {test_result['accuracy']:.2f}%")

                # ---------- W&B LOGGING ----------
                if self.use_wandb and self.wandb_run is not None:
                    self.wandb_run.log({
                        "epoch": epoch + 1,
                        "train/loss": train_result["loss"][0],
                        "train/acc": train_result["accuracy"],
                        "val/loss": val_result["loss"][0],
                        "val/acc": val_result["accuracy"],
                        "test/loss": test_result["loss"][0],
                        "test/acc": test_result["accuracy"],
                        "lr": self.optimizer.param_groups[0]["lr"]
                    }, step=epoch + 1)

                # Early Stopping and Saving (ONLY Master Rank)
                if best_acc is None or val_result["accuracy"] > best_acc:
                    best_acc = val_result["accuracy"]
                    checkpoint_test_acc = test_result["accuracy"]
                    epochs_without_improvement = 0
                    if checkpoint_path:
                        torch.save({
                            # CRITICAL: Save self.model.module for DDP
                            "model_state": self.model.module.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "best_acc": best_acc,
                            "checkpoint_test_acc": checkpoint_test_acc,
                            "ewi": epochs_without_improvement
                        }, checkpoint_path)
                        print(f"*** Saved checkpoint at epoch {epoch + 1}")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement % update_lr_epoch_num == 0:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= lr_factor
                        print(f"Reducing learning rate to {self.optimizer.param_groups[0]['lr']:.6e}")
                    if early_stopping and epochs_without_improvement >= early_stopping:
                        print(f"*** Early stopping at epoch {epoch + 1} ***")
                        break

            # Synchronize processes before next epoch (keeps GPUs in sync)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        # Final Evaluation logic (Confusion Matrix)
        if self.is_master:
            self._finalize_results(dl_test, checkpoint_path)

        return TrainingResults(train_loss, val_loss, test_loss, train_acc, val_acc, test_acc)

    def _finalize_results(self, dl_test, checkpoint_path):
        """Helper to generate final reports - Rank 0 only"""
        y_true, y_pred = [], []
        self.model.eval()
        with torch.no_grad():
            for X, y, _ in dl_test:
                preds = self.model(X.to(self.device)).argmax(dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)

        if checkpoint_path:
            results_dir = os.path.join(os.path.dirname(checkpoint_path), "results")
            os.makedirs(results_dir, exist_ok=True)

            # --- Save Confusion Matrix ---
            cm_df = pd.DataFrame(cm, index=["No Fire", "Fire"], columns=["Pred No Fire", "Pred Fire"])
            cm_df.to_csv(os.path.join(results_dir, "confusion_matrix.csv"))

            # --- ADD THESE 4 LINES FOR THE REPORT ---
            report = classification_report(y_true, y_pred, target_names=["No Fire", "Fire"], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(results_dir, "classification_report.csv"))
            print("\n*** Classification Report ***\n", report_df)

    def _run_epoch(self, dl, train, max_batches=None):
        total_loss, total_correct, total_samples = 0.0, 0, 0
        desc = "Training" if train else "Evaluating"

        loop = tqdm(enumerate(dl), total=len(dl), desc=desc, disable=not self.is_master)

        for i, batch in loop:
            if max_batches is not None and i >= max_batches:
                break

            batch_result = self.train_batch(batch) if train else self.test_batch(batch)
            batch_size = len(batch[1])

            total_loss += batch_result["loss"] * batch_size
            total_correct += batch_result["accuracy"] * batch_size
            total_samples += batch_size

        # ---- Syncrnize between the GPUs ----
        metrics = torch.tensor([total_loss, total_correct, total_samples], device=self.device)

        # Sum all the metrics from the GPUs
        metrics = self.all_reduce_tensor(metrics)

        # Extract the syncronized values
        global_loss, global_correct, global_samples = metrics[0].item(), metrics[1].item(), metrics[2].item()

        return {
            "loss": [global_loss / global_samples],
            "accuracy": 100 * global_correct / global_samples
        }

    def all_reduce_tensor(self, tensor, op=torch.distributed.ReduceOp.SUM):
        """Synchronizes a tensor across all GPUs by summing its values."""
        if not torch.distributed.is_initialized():
            return tensor

        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=op)
        return rt


class ClassificationGuidedEncoding(Trainer):
    def train_batch(self, batch):
        X, y, _ = batch
        X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        self.optimizer.zero_grad()
        predictions = self.model(X)
        loss = self.loss_fn(predictions, y)
        loss.backward()
        self.optimizer.step()

        num_correct = (predictions.argmax(dim=1) == y).sum().item()
        return {"loss": loss.item(), "accuracy": num_correct / len(y)}

    def test_batch(self, batch):
        X, y, _ = batch
        X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        with torch.no_grad():
            predictions = self.model(X)
            loss = self.loss_fn(predictions, y)
            num_correct = (predictions.argmax(dim=1) == y).sum().item()
        return {"loss": loss.item(), "accuracy": num_correct / len(y)}
