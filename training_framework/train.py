import os
import sys
import yaml
import argparse
import logging
import shutil
import random
from datetime import datetime
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import wandb

from functions.training import ClassificationGuidedEncoding
from functions.plot import plot_fit
from functions.datasets import prepare_dataloaders
from collections import Counter

sys.path.append(os.path.abspath(".."))

def analyze_class_distribution(train_loader, val_loader, test_loader, is_master=True):
    train_counts = Counter(train_loader.dataset.labels)
    val_counts = Counter(val_loader.dataset.labels)
    test_counts = Counter(test_loader.dataset.labels)

    if is_master:
        print("\nTrain class distribution:")
        print(train_counts)

        print("\nValidation class distribution:")
        print(val_counts)

        print("\nTest class distribution:")
        print(test_counts)

    class0_train_val = train_counts.get(0, 0) + val_counts.get(0, 0)
    class1_train_val = train_counts.get(1, 0) + val_counts.get(1, 0)
    total_train_val = len(train_loader.dataset.labels) + len(val_loader.dataset.labels)

    return class0_train_val, class1_train_val, total_train_val

# ===================== Reproducibility =====================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ===================== Main =====================
def main(config_path: str):

    # Get environment variables set by torchrun
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize the "handshake" between GPUs
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Only the first process (rank 0) does the "office work"
    is_master = (rank == 0)

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

    if is_master:
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Save config copy
        shutil.copy(config_path, os.path.join(experiment_dir, "config.yaml"))

    # ---------- Logging ----------
    log_path = os.path.join(logs_dir, "train.log")

    if is_master:
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
    else:
        logging.basicConfig(level=logging.ERROR)

    # ---------- Reproducibility ----------
    seed = config.get("seed", 42)
    set_seed(seed)
    if is_master:
        logging.info(f"Seed set to: {seed}")
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

    if is_master:
        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=f"{timestamp}_{model_name}",
            config=config,
        )
    else:
        run = None  # Other GPUs don't need a WandB handle

    # ---------- Prepare Dataloaders ----------
    train_loader, val_loader, test_loader, num_classes = prepare_dataloaders(
        image_size=image_size,
        images_dir=images_dir,
        labels_csv_path=labels_csv_path,
        batch_size=batch_size,
        config=config,
        rank=rank,
        world_size=world_size
    )

    class0_tv, class1_tv, total_tv = analyze_class_distribution(
        train_loader, val_loader, test_loader, is_master
    )
    training_weight = [class1_tv/total_tv , class0_tv/total_tv]

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

    # Optionally freeze backbone — only train the classification head
    if config["training"].get("freeze_backbone", False):
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze classification head
        if "resnet" in model_name.lower():
            for param in model.fc.parameters():
                param.requires_grad = True
        elif "vgg" in model_name.lower() or "alexnet" in model_name.lower():
            for param in model.classifier[-1].parameters():
                param.requires_grad = True
        elif "densenet" in model_name.lower():
            for param in model.classifier.parameters():
                param.requires_grad = True
        if is_master:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logging.info(f"Backbone frozen. Training {trainable:,} / {total:,} parameters (head only)")

    model = model.to(device)

    # Wrap model to synchronize gradients across GPUs
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # Syncs Batch Norm layers
    model = DDP(model, device_ids=[rank])

    if is_master:
        logging.info(f"Number of classes: {num_classes}")
        logging.info(f"Model: {model_name}")
        logging.info(f"Learning rate: {lr}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Epochs: {num_epochs}")

    # ---------- Loss & Optimizer ----------
    weights = torch.tensor(training_weight, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    # ---------- Trainer ----------
    trainer = ClassificationGuidedEncoding(
        model,
        loss_fn,
        optimizer,
        use_wandb=is_master,
        wandb_run=run,
        device=device,
        is_master=is_master,
    )

    # ---------- Training ----------
    checkpoint_path = os.path.join(checkpoints_dir, config["checkpoint"]["path"])

    fit_res = trainer.fit(
        train_loader,
        val_loader,
        test_loader,
        config=config,
        checkpoints=checkpoint_path,
    )

    if is_master:
        logging.info("Training completed successfully.")

        # ---------- Save Plot ----------
        fig, axes = plot_fit(fit_res)
        plot_path = os.path.join(plots_dir, "fit.png")
        fig.savefig(plot_path)
        logging.info(f"Saved training plot to {plot_path}")

        # ---------- Save Best Metrics to W&B ----------
        if fit_res.val_acc:
            best_val_idx = int(np.argmax(fit_res.val_acc))
            run.summary["best_val_acc"] = fit_res.val_acc[best_val_idx]
            run.summary["best_test_acc"] = fit_res.test_acc[best_val_idx]

        run.finish()
        logging.info("W&B run finished.")

    # Wait for master to finish plots and reports
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

# ===================== Entry Point =====================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

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