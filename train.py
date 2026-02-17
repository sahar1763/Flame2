import os
import sys
import yaml
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import wandb

import logging
from datetime import datetime
import shutil

from functions.training import ClassificationGuidedEncoding
from functions.plot import plot_fit
from functions.datasets import prepare_dataloaders

sys.path.append(os.path.abspath('.'))

def main(config_path: str):
    # Load config file
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # --- Dataset paths & params ---
    images_dir = config["dataset"]["images_dir"]
    labels_csv_path = config["dataset"]["labels_csv"]
    image_size = config["dataset"]["image_size"]

    # --- Training hyperparameters ---
    model_name = config["training"]["model_name"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    early_stopping = config["training"]["early_stopping"]
    print_every = config["training"]["print_every"]
    max_batches_per_epoch = config["training"]["max_batches_per_epoch"]

    # --- W&B params ---
    wandb_entity = config["wandb"]["entity"]
    wandb_project = config["wandb"]["project"]

    # --- Checkpoint path ---
    checkpoint_path = config["checkpoints"]["path"]

    # ---------- W&B INIT ----------
    run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        config={
            "model": model_name,
            "phase": "Phase2_RGB",
            "image_size": image_size,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "optimizer": "Adam",
            "loss": "CrossEntropy",
        },
    )

    # ---------- Prepare Dataloaders ----------
    train_loader, val_loader, test_loader, num_classes = prepare_dataloaders(
        image_size=image_size,
        images_dir=images_dir,
        labels_csv_path=labels_csv_path,
        batch_size=batch_size
    )

    # ---------- Initialize Model ----------
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    resnet = resnet.to(device)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize Trainer
    trainer = ClassificationGuidedEncoding(
        resnet,
        loss_fn,
        optimizer,
        use_wandb=True,
        wandb_run=run,
        device=device
    )

    # ---------- Training ----------
    fig_optim = None
    fit_res = trainer.fit(
        train_loader,
        val_loader,
        test_loader,
        num_epochs=num_epochs,
        checkpoints=checkpoint_path,
        early_stopping=early_stopping,
        print_every=print_every,
        max_batches_per_epoch=max_batches_per_epoch
    )

    fig, axes = plot_fit(fit_res, fig=fig_optim)

    print(f"Training Complete! Model saved at {checkpoint_path}.pt")

    # Save best metrics to W&B
    if fit_res.val_acc:
        run.summary["best_val_acc"] = max(fit_res.val_acc)
    if fit_res.test_acc:
        run.summary["best_test_acc"] = max(fit_res.test_acc)

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train wildfire detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="configTrainModel.yaml",
        help="Path to YAML config file containing all hyperparameters and paths"
    )
    args = parser.parse_args()
    print(f"Using config file: {args.config}")

    main(args.config)

# python train.py --config config.yaml
# Linux: python train.py --config /home/user/project/configs/exp1.yaml
# Windows: python train.py --config C:\Users\user\project\configs\exp1.yaml