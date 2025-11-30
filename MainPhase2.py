# Standard library
import os
import sys
import yaml

# Third-party libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import wandb

from functions.training import ClassificationGuidedEncoding
from functions.plot import plot_fit
from functions.datasets import prepare_dataloaders

sys.path.append(os.path.abspath('.'))


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Paths (replace with actual paths)
    images_dir = 'Datasets_FromDvir/Datasets/rgb_images'
    labels_excel_path = 'Datasets_FromDvir/Datasets/labels.csv'

    # Parameter Initialization
    model_name = "resnet18"

    lr = 1e-4
    weight_decay = 1e-4
    batch_size = 100
    num_epochs = 2

    early_stopping = 10
    print_every = 1
    max_batches_per_epoch = 100

    path = r"C:\Projects\Flame2\wildfire_detector\config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---------- W&B INIT ----------
    run = wandb.init(
        entity="WildFire_Detector",       # your team/workspace
        project="Phase2_ResNet18",        # project for this model
        config={
            "model": model_name,
            "phase": "Phase2_RGB",
            "image_size": config["phase2"]["net_image_size"],
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "optimizer": "Adam",
            "loss": "CrossEntropy",
        },
    )

    train_loader, val_loader, test_loader, num_classes = prepare_dataloaders(
        image_size=config["phase2"]["net_image_size"], # TODO: Check and validate image size
        images_dir='Datasets_FromDvir/Datasets/rgb_images',
        labels_csv_path='Datasets_FromDvir/Datasets/labels.csv',
        batch_size=batch_size
    )

    
    # Initialize Model, Loss, and Optimizer

    # Use pretrained ResNet18 and adjust the last layer
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    resnet = resnet.to(device)
    
    # Define Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Initialize Trainer
    trainer = ClassificationGuidedEncoding(resnet, loss_fn, optimizer, use_wandb=True, wandb_run=run, device = device)
    
    # Training the Model
    fig_optim = None
    fit_res = trainer.fit(
        train_loader, val_loader, test_loader,
        num_epochs=num_epochs,
        checkpoints="resnet_fire_classifier",
        early_stopping=early_stopping,
        print_every=print_every,
        max_batches_per_epoch=max_batches_per_epoch
    )
    
    fig, axes = plot_fit(fit_res, fig=fig_optim)
    
    print("Training Complete! Model saved at resnet_fire_classifier.pt")

    if fit_res.val_acc:
        run.summary["best_val_acc"] = max(fit_res.val_acc)
    if fit_res.test_acc:
        run.summary["best_test_acc"] = max(fit_res.test_acc)

    run.finish()

if __name__ == "__main__":
    main()

