# Standard library
import os
import random
import sys

# Third-party libraries
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler


sys.path.append(os.path.abspath(''))



# Custom dataset from pre-split lists
class FireSmokeDatasetFromLists(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label, image_path
        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            return torch.zeros(3, 224, 224), 0, "error" # TODO


def load_image_label_data(images_dir, labels_excel_path):
    df = pd.read_csv(labels_excel_path)

    # Generate full image paths
    df['image_path'] = df['id'].apply(lambda x: os.path.join(images_dir, x))
    # Filter out missing or invalid image files (non-image, wrong extension)
    df = df[df['image_path'].apply(lambda p: os.path.isfile(p) and p.lower().endswith(('.jpg', '.jpeg', '.png')))]

    # Keep rows where 'fire' is not NaN
    df = df.dropna(subset=['fire'], how='all')

    # Remove rows where fire has invalid values (not 0 or 1), and convert valid ones to int
    df = df[
        ((df['fire'].isin([0, 1])) | df['fire'].isna())
        ]

    # Fill missing values with 0
    df['fire'] = df['fire'].fillna(0).astype(int)

    # Binary label: 1 = Fire , 0 = No Fire
    def map_label(row):
        return 1 if row['fire'] == 1 else 0

    df['label'] = df.apply(map_label, axis=1)

    # Generate full image paths
    df['image_path'] = df['id'].apply(lambda x: os.path.join(images_dir, x))

    # Filter out missing files
    df = df[df['image_path'].apply(os.path.exists)]

    image_paths = df['image_path'].tolist()
    labels = df['label'].tolist()
    return image_paths, labels


def prepare_dataloaders(image_size, images_dir, labels_csv_path, batch_size, config, rank=0, world_size=1):
    # Set random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load CSV
    df = pd.read_csv(labels_csv_path)
    df['image_path'] = df['id'].apply(lambda x: os.path.join(images_dir, x))
    df = df[df['image_path'].apply(os.path.exists)]

    # Map label
    df['fire'] = df['fire'].fillna(0).astype(int)
    df['label'] = df.apply(lambda row: 1 if row['fire'] == 1 else 0, axis=1)

    # ---------- Handle test-only datasets ----------
    test_only_sources = config.get("dataset", {}).get("test_only", [])
    test_df = df[df['dataset'].isin(test_only_sources)]
    remaining_df = df[~df['dataset'].isin(test_only_sources)]

    # Split remaining into train/val
    val_ratio = config["dataset"]["val_ratio"]
    train_df, val_df = train_test_split(
        remaining_df, test_size=val_ratio, stratify=remaining_df['label'], random_state=random_seed
    )

    # Calculate this just to know the number of output nodes
    num_classes = len(np.unique(train_df['label']))

    # Create datasets
    train_dataset = FireSmokeDatasetFromLists(train_df['image_path'].tolist(), train_df['label'].tolist(), transform=train_transform)
    val_dataset = FireSmokeDatasetFromLists(val_df['image_path'].tolist(), val_df['label'].tolist(), transform=test_transform)
    test_dataset = FireSmokeDatasetFromLists(test_df['image_path'].tolist(), test_df['label'].tolist(), transform=test_transform)

    # Distributed Sampler handles splitting data across the 8 GPUs
    # Note: Using standard DistributedSampler here.
    # For weighted distributed sampling, specialized custom classes are usually needed.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    # We also create samplers for validation and test to ensure consistent data handling
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    print(f"world_size: {world_size}")
    print(f"rank: {rank}")

    # Create loaders
    dataloader_params = config.get("dataloader", {})
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, **dataloader_params)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, **dataloader_params)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, **dataloader_params)

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)} (includes test-only datasets)")

    return train_loader, val_loader, test_loader, num_classes
