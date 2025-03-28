import time

import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.nn.functional as F
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
        model.to(self.device)

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
            print(f"*** Loading checkpoint file")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            best_acc = checkpoint.get("best_acc", best_acc)
            epochs_without_improvement = checkpoint.get("ewi", epochs_without_improvement)

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

            # Testing
            test_result = self._run_epoch(dl_test, train=False, max_batches=max_batches_per_epoch)
            test_loss.extend(test_result["loss"])
            test_acc.append(test_result["accuracy"])

            # Print statistics
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                print(f"--- EPOCH {epoch + 1}/{num_epochs} ---")
                print(f"  Train Loss: {sum(train_result['loss']) / len(train_result['loss']):.4f}")
                print(f"  Val Loss: {sum(val_result['loss']) / len(val_result['loss']):.4f}")
                print(f"  Test Loss: {sum(test_result['loss']) / len(test_result['loss']):.4f}")

            # Early Stopping Check
            if best_acc is None or val_result["accuracy"] > best_acc:
                best_acc = val_result["accuracy"]
                epochs_without_improvement = 0
                if checkpoint_path:
                    torch.save({"model_state": self.model.state_dict(), "best_acc": best_acc, "ewi": epochs_without_improvement}, checkpoint_path)
                    print(f"*** Saved checkpoint at epoch {epoch+1}")
            else:
                epochs_without_improvement += 1
                # Reduce Learning Rate by factor of 2 when no improvement in 3 epochs
                if epochs_without_improvement % 3 == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print(f"Reducing learning rate to {param_group['lr']:.6e}")
                if early_stopping and epochs_without_improvement >= early_stopping:
                    print(f"*** Early stopping {checkpoint_path} at epoch {epoch + 1} ***")
                    break

        
        # Return results
        return TrainingResults(
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=test_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            test_acc=test_acc
        )

    # def _run_epoch(self, dl, train):
    #     total_loss, total_acc = [], 0
    #     for batch in dl:
    #         batch_result = self.train_batch(batch) if train else self.test_batch(batch)
    #         total_loss.append(batch_result["loss"])
    #         total_acc += batch_result["accuracy"]
    #     return {"loss": total_loss, "accuracy": 100*total_acc / len(dl)}
    def _run_epoch(self, dl, train, max_batches=None):
        total_loss, total_acc = [], 0
        for i, batch in enumerate(dl):
            if i%50==0:
                print(i)
            if max_batches is not None and i >= max_batches:
                break
            batch_result = self.train_batch(batch) if train else self.test_batch(batch)
            total_loss.append(batch_result["loss"])
            total_acc += batch_result["accuracy"]
    
        return {
            "loss": total_loss,
            "accuracy": 100 * total_acc / len(total_loss)
        }
    # def _run_epoch(self, dl, train, max_batches=None):
    #     total_loss, total_acc = [], 0
    
    #     for i, batch in enumerate(dl):
    #         start = time.time()
    
    #         if i % 10 == 0:
    #             print(f"[{i}] Running batch {i}/{len(dl)}...")
    
    #         if max_batches is not None and i >= max_batches:
    #             break
    
    #         batch_result = self.train_batch(batch) if train else self.test_batch(batch)
    
    #         batch_time = time.time() - start
    #         print(f"    Batch {i} done in {batch_time:.2f} sec | Loss: {batch_result['loss']:.4f} | Acc: {batch_result['accuracy']:.2%}")
    
    #         total_loss.append(batch_result["loss"])
    #         total_acc += batch_result["accuracy"]
    
    #     return {
    #         "loss": total_loss,
    #         "accuracy": 100 * total_acc / len(total_loss)
    #     }





class ClassificationGuidedEncoding(Trainer):
    """
    Trainer for our Classification-Guided Encoding model.
    """

    def __init__(
        self,
        model: AutoencoderClassifier,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device="cpu",
    ):

        super().__init__(model, loss_fn, optimizer, device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        model = model.to(self.device)

    def train_batch(self, batch):
        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)

        # Train the model on one batch of data.
        # Forward pass: Compute predictions
        self.optimizer.zero_grad()
        
        predictions = self.model(X)

        # Compute loss
        loss = self.loss_fn(predictions, y)
        
        # Backward pass
        loss.backward()
        
        # Optimize parameters: Update model parameters
        self.optimizer.step()
        
        # Calculate the number of correct predictions
        predicted_labels = torch.argmax(predictions, dim=1)
        num_correct = (predicted_labels == y).sum().item()  # Convert to int

        # Convert loss to a scalar value for returning
        batch_loss = loss.item()

        return {"loss": batch_loss, "accuracy": num_correct / len(y)}

    def test_batch(self, batch):
        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            # Evaluate the model on one batch of data.
            # Forward pass: Compute predictions
            predictions = self.model(X)
            # Compute loss
            loss = self.loss_fn(predictions, y)
            # Calculate the number of correct predictions
            predicted_labels = torch.argmax(predictions, dim=1)
            num_correct = (predicted_labels == y).sum().item()  # Convert to int
            # Convert loss to a scalar value for returning
            batch_loss = loss.item()

        return {"loss": batch_loss, "accuracy": num_correct / len(y)}

