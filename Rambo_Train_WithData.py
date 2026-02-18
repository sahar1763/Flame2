import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import time

# --- Configuration ---
CSV_FILE = 'labels.csv'
IMAGE_FOLDER = 'Rambo_Dataset'  # Updated to match your folder name
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 5


# --- 1. Custom Dataset with Auto-Filtering ---
class FireDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 1. Load the full CSV
        full_df = pd.read_csv(csv_file)

        # 2. Get list of files actually present on disk
        if not os.path.exists(root_dir):
            print(f"ERROR: Folder '{root_dir}' not found!")
            self.annotations = pd.DataFrame()
            return

        available_files = set(os.listdir(root_dir))

        # 3. Filter the CSV to keep ONLY the available files
        self.annotations = full_df[full_df['id'].isin(available_files)].reset_index(drop=True)

        print(f"Original CSV has {len(full_df)} rows.")
        print(f"Found {len(available_files)} images in '{root_dir}'.")
        print(f"Training will run on the matching {len(self.annotations)} images.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index]['id']
        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations.iloc[index]['fire'])

        if self.transform:
            image = self.transform(image)

        return image, label


# --- 2. Setup Training ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Standard ResNet transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize Dataset
    dataset = FireDataset(csv_file=CSV_FILE, root_dir=IMAGE_FOLDER, transform=transform)

    # Safety check: Do we have data?
    if len(dataset) == 0:
        print("ERROR: No matching images found! Check your filenames and folder name.")
        return

    # For mini-test, use all data for training
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load Model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training Loop ---
    print("\n--- Starting Mini-Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Loss: {running_loss / len(train_loader):.4f} | Accuracy: {epoch_acc:.2f}%")

    print("\nTest complete!")

    # --- 4. SAVING THE WEIGHTS ---
    SAVE_PATH = "fire_model.pth"
    print(f"Saving weights to {SAVE_PATH}...")
    torch.save(model.state_dict(), SAVE_PATH)
    print("Done! You can now see the file by typing 'ls -lh' in the terminal.")


if __name__ == '__main__':
    main()