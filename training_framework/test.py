import os
import argparse
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np
import yaml

from functions.datasets import prepare_dataloaders  # שימוש בקוד שלך

def evaluate_model(model, dataloader, device, label_names, save_dir):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    image_names = []

    with torch.no_grad():
        for batch_idx, (x, y, names) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())  # סיכוי ל"Fire" (class 1)
            image_names.extend(names)

    # Accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nAccuracy: {accuracy*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=["No Fire", "Fire"], columns=["Pred No Fire", "Pred Fire"])
    print("\nConfusion Matrix:")
    print(cm_df)

    cm_csv_path = os.path.join(save_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)
    print(f"Saved confusion matrix to {cm_csv_path}")

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=label_names, output_dict=True, digits=3)
    report_df = pd.DataFrame(report).transpose()
    print("\n*** Classification Report ***")
    print(report_df)

    report_csv_path = os.path.join(save_dir, "classification_report.csv")
    report_df.to_csv(report_csv_path)
    print(f"Saved classification report to {report_csv_path}")

    # Save predictions to CSV
    df_preds = pd.DataFrame({
        "image": image_names,
        "true_label": all_labels,
        "pred_label": all_preds,
        "prob_fire": all_probs
    })
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "predictions.csv")
    df_preds.to_csv(csv_path, index=False)
    print(f"\nSaved predictions to {csv_path}")

    # ROC curve + AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    roc_path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve to {roc_path}")

    return accuracy, cm, roc_auc

def main(experiment_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = os.path.join(experiment_path, "checkpoints", "best_model.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config_path = os.path.join(experiment_path, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_name = config["training"]["model_name"]
    num_classes = 2

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

    model.load_state_dict(checkpoint["model_state"])

    image_size = config["dataset"]["image_size"]
    images_dir = config["dataset"]["images_dir"]
    labels_csv_path = config["dataset"]["labels_csv"]
    batch_size = config["training"]["batch_size"]

    _, _, test_loader, _ = prepare_dataloaders(
        image_size=image_size,
        images_dir=images_dir,
        labels_csv_path=labels_csv_path,
        batch_size=batch_size,
        config=config
    )

    label_names = ["No Fire", "Fire"]
    results_dir = os.path.join(experiment_path, "test")
    os.makedirs(results_dir, exist_ok=True)
    evaluate_model(model, test_loader, device, label_names, results_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate wildfire detection model")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Path to experiment folder containing checkpoints/config.yaml")
    args = parser.parse_args()
    main(args.experiment)

    # python3 test.py --experiment ../experiments/2026-03-11_15-27-18_resnet18