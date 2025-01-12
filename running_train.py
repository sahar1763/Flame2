import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from models import (
    Logistic_two_stream, Flame_one_stream, VGG16, Vgg_two_stream, Logistic, Flame_two_stream,
    Mobilenetv2, Mobilenetv2_two_stream, LeNet5_one_stream, LeNet5_two_stream, Resnet18, Resnet18_two_stream
)

# === הגדרת פונקציות עיבוד והכנה ===

# שלב 1: עיבוד תמונה בודדת
def preprocess_image(image_path, target_size):
    image = Image.open(image_path).convert('RGB')  # פתיחת התמונה
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),  # שינוי גודל
        transforms.ToTensor(),  # המרה ל-Tensor
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # נרמול
    ])
    return transform(image).unsqueeze(0)  # הוספת ממד Batch


# שלב 2: עיבוד קבוצה של תמונות מסודרות בתיקיות
def evaluate_model_on_dataset(model, dataset_path, target_size):
    # עיבוד התמונות
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),  # שינוי גודל
        transforms.ToTensor(),  # המרה ל-Tensor
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # נרמול
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
            outputs = model(inputs, None, mode='rgb')  # או 'ir'/'both' לפי הצורך
            predicted = torch.argmax(outputs, dim=0)

            # בדיקה אם צדק
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # חישוב הדיוק
    accuracy = correct / total * 100
    print(f"Accuracy on dataset: {accuracy:.2f}%")
    return accuracy


# === פונקציה להדפסת דוגמאות ===

def display_predictions(model, dataloader, class_names, class_labels, num_samples=5):
    """הדפסת דוגמאות של תמונות עם המחלקות שנחזו והאמיתיות."""
    model.eval()
    samples_shown = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            if samples_shown >= num_samples:
                break

            # שליחת התמונות למודל
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs, None, mode='rgb')  # שימוש במודל
            predicted = torch.argmax(outputs, dim=0)

            # הצגת תמונה עם המחלקות
            for i in range(len(inputs)):
                if samples_shown >= num_samples:
                    break

                img = inputs[i].cpu().permute(1, 2, 0).numpy()  # המרה למטריצה להצגה
                img = (img * 0.5) + 0.5  # ביטול הנרמול
                true_label = class_labels[labels.item()]  # מיפוי תווית אמיתית
                predicted_label = class_labels[predicted.item()]  # מיפוי תווית מנובאת
                plt.imshow(img)
                plt.title(f"True: {true_label}, Predicted: {predicted_label}")
                plt.axis('off')
                plt.show()
                samples_shown += 1



# === הגדרות כלליות ===
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # בחירת GPU אם קיים
target_size = 254  # גודל התמונות

# === מודל 1: בדיקה על תמונה בודדת ===
# הגדרת המודל וטעינת המשקולות
model_path = './saved_models/Flame_one_stream_rgb_index0_epoch35_final.pth'
model = Flame_one_stream().to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# עיבוד תמונה בודדת
#rgb_image_path = './P1.jpg'
#rgb_tensor = preprocess_image(rgb_image_path, target_size).to(device)

# חיזוי
#with torch.no_grad():
 #   output = model(rgb_tensor, None, mode='rgb')  # או 'rgb'/'ir' בהתאם למודל
 #   predicted_class = torch.argmax(output, dim=0).item()
 #   print(f"The predicted class for the image is: {predicted_class}")

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
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # נרמול
])

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# שמות המחלקות
#class_names = dataset.classes  # שמות המחלקות מהתיקיות

# הדפסת דוגמאות
display_predictions(model, dataloader, dataset.classes, class_labels, num_samples=20)
