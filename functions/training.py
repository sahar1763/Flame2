import time
import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from collections import namedtuple

from .Classifier import Classifier, AutoencoderClassifier
from .autoencoder import Encoder2

# Creating a class to store the results
TrainingResults = namedtuple('TrainingResults', ['train_loss', 'val_loss', 'test_loss', 'train_acc', 'val_acc', 'test_acc'])


class Trainer:
    def __init__(self, model, loss_fn, optimizer, device="cpu"):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train_batch(self, batch):
        raise NotImplementedError()

    def test_batch(self, batch):
        raise NotImplementedError()

    def fit(self, dl_train, dl_val, dl_test, num_epochs, checkpoints=None, early_stopping=None, print_every=1, max_batches_per_epoch=None):
        best_acc = None
        epochs_without_improvement = 0
        checkpoint_path = f"{checkpoints}.pt" if checkpoints else None

        train_loss, train_acc = [], []
        val_loss, val_acc = [], []
        test_loss, test_acc = [], []

        # Load checkpoint if it exists
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print(f"*** Loading checkpoint file from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state"])
                best_acc = checkpoint.get("best_acc", best_acc)
                epochs_without_improvement = checkpoint.get("ewi", epochs_without_improvement)
            except Exception as e:
                print(f"Failed loading checkpoint due to: {e}")

        # Train loop
        print("Start training")
        for epoch in range(num_epochs):
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

            # Print statistics
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                print(f"--- EPOCH {epoch + 1}/{num_epochs} ---")
                print(f"  Train Loss: {train_result['loss'][0]:.4f} | Train Acc: {train_result['accuracy']:.2f}%")
                print(f"  Val Loss: {val_result['loss'][0]:.4f} | Val Acc: {val_result['accuracy']:.2f}%")
                print(f"  Test Loss: {test_result['loss'][0]:.4f} | Test Acc: {test_result['accuracy']:.2f}%")

            # Early Stopping Check
            if best_acc is None or val_result["accuracy"] > best_acc:
                best_acc = val_result["accuracy"]
                epochs_without_improvement = 0
                if checkpoint_path:
                    torch.save({"model_state": self.model.state_dict(), "best_acc": best_acc, "ewi": epochs_without_improvement}, checkpoint_path)
                    print(f"*** Saved checkpoint at epoch {epoch+1}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement % 3 == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                        new_lr = param_group['lr']
                    print(f"Reducing learning rate to {new_lr:.6e}")
                if early_stopping and epochs_without_improvement >= early_stopping:
                    print(f"*** Early stopping at epoch {epoch + 1} ***")
                    break

        return TrainingResults(
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=test_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            test_acc=test_acc
        )

    def _run_epoch(self, dl, train, max_batches=None):
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for i, batch in enumerate(dl):
            if max_batches is not None and i >= max_batches:
                break
            if i % 10 == 0:
                print(f"{'Training' if train else 'Evaluating'} batch {i}/{len(dl)}...")

            batch_result = self.train_batch(batch) if train else self.test_batch(batch)
            batch_size = len(batch[1])

            total_loss += batch_result["loss"] * batch_size
            total_correct += batch_result["accuracy"] * batch_size
            total_samples += batch_size

        return {
            "loss": [total_loss / total_samples],
            "accuracy": 100 * total_correct / total_samples
        }


class ClassificationGuidedEncoding(Trainer):
    def train_batch(self, batch):
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        predictions = self.model(X)
        loss = self.loss_fn(predictions, y)
        loss.backward()
        self.optimizer.step()

        num_correct = (predictions.argmax(dim=1) == y).sum().item()
        return {"loss": loss.item(), "accuracy": num_correct / len(y)}

    def test_batch(self, batch):
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)
        with torch.no_grad():
            predictions = self.model(X)
            loss = self.loss_fn(predictions, y)
            num_correct = (predictions.argmax(dim=1) == y).sum().item()
        return {"loss": loss.item(), "accuracy": num_correct / len(y)}
