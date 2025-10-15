# Standard library
import os
import sys
import yaml

# Third-party libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

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


    path = r"C:\Projects\Flame2\wildfire_detector\config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, test_loader, num_classes = prepare_dataloaders(
        image_size=config["phase2"]["net_image_size"], # TODO: Check and validate image size
        images_dir='Datasets_FromDvir/Datasets/rgb_images',
        labels_csv_path='Datasets_FromDvir/Datasets/labels.csv',
        batch_size=100
    )

    
    # Initialize Model, Loss, and Optimizer

    # Use pretrained ResNet18 and adjust the last layer
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    resnet = resnet.to(device)
    
    # Define Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # Initialize Trainer
    trainer = ClassificationGuidedEncoding(resnet, loss_fn, optimizer, device)
    
    # Training the Model
    fig_optim = None
    fit_res = trainer.fit(
        train_loader, val_loader, test_loader,
        num_epochs=2,
        checkpoints="resnet_fire_classifier",
        early_stopping=10,
        print_every=1,
        max_batches_per_epoch=100
    )
    
    fig, axes = plot_fit(fit_res, fig=fig_optim)
    
    print("Training Complete! Model saved at resnet_fire_classifier.pt")

if __name__ == "__main__":
    main()

