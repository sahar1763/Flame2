import os
import sys
import yaml
import argparse
import logging
import shutil
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
import wandb

from functions.training import ClassificationGuidedEncoding
from functions.plot import plot_fit
from functions.datasets import prepare_dataloaders

sys.path.append(os.path.abspath(".."))


# ===================== Reproducibility =====================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================== Main =====================
def main(config_path: str):

    # ---------- Load Config ----------
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---------- Create Unique Experiment Folder ----------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = config["training"]["model_name"]

    experiment_dir = os.path.join("../experiments", f"{timestamp}_{model_name}")
    logs_dir = os.path.join(experiment_dir, "logs")
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    plots_dir = os.path.join(experiment_dir, "plots")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Save config copy
    shutil.copy(config_path, os.path.join(experiment_dir, "config.yaml"))

    # ---------- Logging ----------
    log_path = os.path.join(logs_dir, "train.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Experiment directory: {experiment_dir}")
    logging.info(f"Using config file: {config_path}")

    # ---------- Reproducibility ----------
    seed = config.get("seed", 42)
    set_seed(seed)
    logging.info(f"Seed set to: {seed}")

    # ---------- Device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ---------- Dataset Params ----------
    images_dir = config["dataset"]["images_dir"]
    labels_csv_path = config["dataset"]["labels_csv"]
    image_size = config["dataset"]["image_size"]

    # ---------- Training Hyperparameters ----------
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]

    # ---------- W&B ----------
    wandb_entity = config["wandb"]["entity"]
    wandb_project = config["wandb"]["project"]

    run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=f"{timestamp}_{model_name}",
        config=config,
    )

    # ---------- Prepare Dataloaders ----------
    train_loader, val_loader, test_loader, num_classes = prepare_dataloaders(
        image_size=image_size,
        images_dir=images_dir,
        labels_csv_path=labels_csv_path,
        batch_size=batch_size,
        config=config
    )

    logging.info(f"Number of classes: {num_classes}")

    # ---------- Initialize Model ----------
    model_name = config["training"]["model_name"]
    model = getattr(models, model_name)(pretrained=True)

    if "resnet" in model_name.lower():
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif "vgg" in model_name.lower() or "alexnet" in model_name.lower():
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    elif "densenet" in model_name.lower():
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    logging.info(f"Model: {model_name}")
    logging.info(f"Learning rate: {lr}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Epochs: {num_epochs}")

    # ---------- Loss & Optimizer ----------
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---------- Trainer ----------
    trainer = ClassificationGuidedEncoding(
        model,
        loss_fn,
        optimizer,
        use_wandb=True,
        wandb_run=run,
        device=device
    )

    # ---------- Training ----------
    checkpoint_path = os.path.join(checkpoints_dir, "best_model")

    fit_res = trainer.fit(
        train_loader,
        val_loader,
        test_loader,
        config=config,
        checkpoints=checkpoint_path,
    )

    logging.info("Training completed successfully.")

    # ---------- Save Plot ----------
    fig, axes = plot_fit(fit_res)
    plot_path = os.path.join(plots_dir, "fit.png")
    fig.savefig(plot_path)
    logging.info(f"Saved training plot to {plot_path}")

    # ---------- Save Best Metrics to W&B ----------
    if fit_res.val_acc:
        run.summary["best_val_acc"] = max(fit_res.val_acc)
    if fit_res.test_acc:
        run.summary["best_test_acc"] = max(fit_res.test_acc)

    run.finish()
    logging.info("W&B run finished.")


# ===================== Entry Point =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train wildfire detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="configTrainModel.yaml",
        help="Path to YAML config file containing all hyperparameters and paths"
    )

    args = parser.parse_args()
    main(args.config)

# python train.py --config config.yaml
# Linux: python train.py --config /home/user/project/configs/exp1.yaml
# Windows: python train.py --config C:\Users\user\project\configs\exp1.yaml