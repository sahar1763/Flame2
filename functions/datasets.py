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


sys.path.append(os.path.abspath('.'))



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
            return torch.zeros(3, 254, 254), 0, "error"


def load_image_label_data(images_dir, labels_excel_path):
    df = pd.read_csv(labels_excel_path)

    # Generate full image paths
    df['image_path'] = df['id'].apply(lambda x: os.path.join(images_dir, x))
    # Filter out missing or invalid image files (non-image, wrong extension)
    df = df[df['image_path'].apply(lambda p: os.path.isfile(p) and p.lower().endswith(('.jpg', '.jpeg', '.png')))]

    # Keep rows where at least one of 'fire' or 'smoke' is not NaN
    df = df.dropna(subset=['fire', 'smoke'], how='all')

    # Remove rows where fire or smoke have invalid values (not 0 or 1), and convert valid ones to int
    df = df[
        ((df['fire'].isin([0, 1])) | df['fire'].isna()) &
        ((df['smoke'].isin([0, 1])) | df['smoke'].isna())
        ]

    # Fill missing values with 0
    df['fire'] = df['fire'].fillna(0).astype(int)
    df['smoke'] = df['smoke'].fillna(0).astype(int)

    # Binary label: 1 = Fire (fire or smoke), 0 = No Fire
    def map_label(row):
        return 1 if row['fire'] == 1 or row['smoke'] == 1 else 0

    df['label'] = df.apply(map_label, axis=1)

    # Generate full image paths
    df['image_path'] = df['id'].apply(lambda x: os.path.join(images_dir, x))

    # Filter out missing files
    df = df[df['image_path'].apply(os.path.exists)]

    image_paths = df['image_path'].tolist()
    labels = df['label'].tolist()
    return image_paths, labels



def prepare_dataloaders(image_size, images_dir, labels_csv_path, batch_size):
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

    # Load paths and labels
    image_paths, labels = load_image_label_data(images_dir, labels_csv_path)

    # Split data
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.4, stratify=labels, random_state=random_seed
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=random_seed
    )

    # Weighted sampling
    class_counts = np.bincount(train_labels)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)

    # Create datasets
    train_dataset = FireSmokeDatasetFromLists(train_paths, train_labels, transform=train_transform)
    val_dataset = FireSmokeDatasetFromLists(val_paths, val_labels, transform=test_transform)
    test_dataset = FireSmokeDatasetFromLists(test_paths, test_labels, transform=test_transform)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              pin_memory=True, prefetch_factor=4, num_workers=12, persistent_workers=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, prefetch_factor=4, num_workers=12, persistent_workers=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=True, prefetch_factor=4, num_workers=12, persistent_workers=True)

    # Optional print
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, len(class_counts)
