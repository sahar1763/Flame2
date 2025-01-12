import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from models import (
    Logistic_two_stream, Flame_one_stream, VGG16, Vgg_two_stream, Logistic, Flame_two_stream,
    Mobilenetv2, Mobilenetv2_two_stream, LeNet5_one_stream, LeNet5_two_stream, Resnet18, Resnet18_two_stream
)

# === הגדרת פונקציות עיבוד והכנה ===


# שלב 2: עיבוד קבוצה של תמונות מסודרות בתיקיות
def evaluate_model_on_dataset(model, dataset_path, target_size):
    # עיבוד התמונות
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),  # שינוי גודל
        transforms.ToTensor(),  # המרה ל-Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # נרמול
    ])

    # טעינת התמונות מהתיקיות
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    print(dataset.class_to_idx)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # בדיקת דיוק על הקבוצה
    correct = 0
    total = 0

    with torch.no_grad():  # מניעת חישובי גרדיאנט
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # חיזוי עם המודל
            outputs = model(inputs)  # או 'ir'/'both' לפי הצורך
            predicted = torch.argmax(outputs, dim=1)

            # בדיקה אם צדק
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # חישוב הדיוק
    accuracy = correct / total * 100
    print(f"Accuracy on dataset: {accuracy:.2f}%")
    return accuracy

def display_predictions(model, dataloader, class_names, class_labels, num_samples=5):
    """Display examples of images with their true and predicted labels."""
    model.eval()
    samples_shown = 0

    # Define the mean and std for denormalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def denormalize(tensor):
        """Denormalize the image tensor."""
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)  # Denormalize
        return tensor

    with torch.no_grad():
        for inputs, labels in dataloader:
            if samples_shown >= num_samples:
                break

            # Send inputs to the correct device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get model predictions
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)

            # Display images and predictions
            for i in range(len(inputs)):
                if samples_shown >= num_samples:
                    break

                img = inputs[i].cpu()  # Move image back to CPU
                img = denormalize(img).permute(1, 2, 0).numpy()  # Denormalize and rearrange dimensions
                img = np.clip(img, 0, 1)  # Clip values to [0, 1] for display

                true_label = class_labels[labels[i].item()]  # Map true label to class name
                predicted_label = class_labels[predicted[i].item()]  # Map predicted label to class name
                plt.imshow(img)
                plt.title(f"True: {true_label}, Predicted: {predicted_label}")
                plt.axis('off')
                plt.show()

                samples_shown += 1



# === הגדרות כלליות ===
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # בחירת GPU אם קיים
target_size = 224  # גודל התמונות

# === מודל 1: בדיקה על תמונה בודדת ===
# הגדרת המודל וטעינת המשקולות
model_path = './saved_models/Resnet18_rgb_index0_epoch35_final.pth'
model = Resnet18(3).to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
# === מודל 2: בדיקה על קבוצה של תמונות ===
# הגדרת הנתיב לתיקיות התמונות
dataset_path = './dataset'  # הנתיב לתיקייה עם התמונות

# חישוב הדיוק על קבוצה של תמונות
evaluate_model_on_dataset(model, dataset_path, target_size)


class_labels = {
    0: 'NN',
    1: 'YY',
    2: 'YN'
}

# עיבוד התמונות
transform = transforms.Compose([
    transforms.Resize((target_size, target_size)),  # שינוי גודל
    transforms.ToTensor(),  # המרה ל-Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # נרמול
])

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# שמות המחלקות
class_names = dataset.classes  # שמות המחלקות מהתיקיות

# הדפסת דוגמאות
display_predictions(model, dataloader, dataset.classes, class_labels, num_samples=10)
