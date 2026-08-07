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


def _collect_predictions(model, dataloader, device):
    """Run the model over a dataloader once and collect aligned arrays."""
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
            all_probs.extend(probs[:, 1].cpu().numpy())  # סיכוי ל"Fire" (class 1)
            image_names.extend(names)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        list(image_names),
    )


def _compute_and_save(all_labels, all_preds, all_probs, image_names,
                      label_names, save_dir, subset_name):
    """Compute metrics for one subset of predictions, save CSVs + ROC.

    Returns a summary dict (one row) for the combined by-dataset table.
    ROC is skipped when the subset contains a single class.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_labels = np.asarray(all_labels)
    all_preds = np.asarray(all_preds)
    all_probs = np.asarray(all_probs)

    # Accuracy
    accuracy = np.mean(all_preds == all_labels) if len(all_labels) else 0.0
    print(f"\n[{subset_name}] Accuracy: {accuracy*100:.2f}%  (n={len(all_labels)})")

    # Confusion Matrix (force both classes so shape is always 2x2)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["No Fire", "Fire"], columns=["Pred No Fire", "Pred Fire"])
    print("\nConfusion Matrix:")
    print(cm_df)
    cm_df.to_csv(os.path.join(save_dir, "confusion_matrix.csv"))

    # Classification report
    report = classification_report(
        all_labels, all_preds, labels=[0, 1],
        target_names=label_names, output_dict=True, digits=3, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(save_dir, "classification_report.csv"))

    # --- Evaluation metrics ---
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    fire_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    fire_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fire_f1 = 2 * fire_precision * fire_recall / (fire_precision + fire_recall) if (fire_precision + fire_recall) > 0 else 0.0
    fpr_val = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr_val = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    print("\n" + "=" * 50)
    print(f"MODEL EVALUATION SUMMARY [{subset_name}]")
    print("=" * 50)
    print(f"  Accuracy:           {accuracy*100:.2f}%")
    print(f"  Fire Precision:     {fire_precision*100:.2f}%")
    print(f"  Fire Recall:        {fire_recall*100:.2f}%")
    print(f"  Fire F1-score:      {fire_f1*100:.2f}%")
    print(f"  False Positive Rate:{fpr_val*100:.2f}%")
    print(f"  False Negative Rate:{fnr_val*100:.2f}%")
    print("=" * 50)

    # ROC curve + AUC (needs both classes present)
    unique_classes = np.unique(all_labels)
    roc_auc = None
    if len(unique_classes) < 2:
        print(f"[{subset_name}] Only one class present -> skipping ROC/AUC.")
    else:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic [{subset_name}]')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, "roc_curve.png"))
        plt.close()

    # Save evaluation summary to CSV
    summary_row = {
        "subset": subset_name,
        "num_images": int(len(all_labels)),
        "accuracy": round(accuracy * 100, 2),
        "fire_precision": round(fire_precision * 100, 2),
        "fire_recall": round(fire_recall * 100, 2),
        "fire_f1": round(fire_f1 * 100, 2),
        "false_positive_rate": round(fpr_val * 100, 2),
        "false_negative_rate": round(fnr_val * 100, 2),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "",
    }
    pd.DataFrame([summary_row]).to_csv(os.path.join(save_dir, "evaluation_summary.csv"), index=False)

    # Save predictions to CSV
    pd.DataFrame({
        "image": image_names,
        "true_label": all_labels,
        "pred_label": all_preds,
        "prob_fire": all_probs,
    }).to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    print(f"[{subset_name}] Saved results to {save_dir}")
    return summary_row


def evaluate_model(model, dataloader, device, label_names, results_dir,
                   images_dir, labels_csv_path):
    """Evaluate the model on the full test loader, saving fused ('all') results
    plus a per-dataset breakdown. Layout (Option B):

        results_dir/
            all/                             fused results
            <dataset>/                       per-dataset results (if >1 dataset)
            evaluation_summary_by_dataset.csv
    """
    all_labels, all_preds, all_probs, image_names = _collect_predictions(model, dataloader, device)

    # Map each image (by basename) -> source dataset using labels.csv
    labels_df = pd.read_csv(labels_csv_path)
    id_to_dataset = dict(zip(labels_df["id"].astype(str), labels_df["dataset"].astype(str)))
    datasets_per_image = np.array([
        id_to_dataset.get(os.path.basename(str(p)), "UNKNOWN") for p in image_names
    ])

    summary_rows = []

    # ----- Fused ('all') results -----
    all_dir = os.path.join(results_dir, "all")
    summary_rows.append(
        _compute_and_save(all_labels, all_preds, all_probs, image_names,
                          label_names, all_dir, "all")
    )

    # ----- Per-dataset results (skip when only one dataset) -----
    unique_datasets = sorted(set(datasets_per_image))
    if len(unique_datasets) > 1:
        for ds in unique_datasets:
            mask = datasets_per_image == ds
            ds_dir = os.path.join(results_dir, ds)
            summary_rows.append(
                _compute_and_save(
                    all_labels[mask], all_preds[mask], all_probs[mask],
                    [image_names[i] for i in np.where(mask)[0]],
                    label_names, ds_dir, ds
                )
            )
    else:
        print(f"[Eval] Single dataset ({unique_datasets}) -> only 'all' results written.")

    # ----- Combined by-dataset summary table -----
    combined_path = os.path.join(results_dir, "evaluation_summary_by_dataset.csv")
    pd.DataFrame(summary_rows).to_csv(combined_path, index=False)
    print(f"\nSaved combined by-dataset summary to {combined_path}")

    return summary_rows

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
    elif "mobilenet" in model_name.lower():
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
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
    evaluate_model(model, test_loader, device, label_names, results_dir,
                   images_dir, labels_csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate wildfire detection model")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Path to experiment folder containing checkpoints/config.yaml")
    args = parser.parse_args()
    main(args.experiment)

    # python3 test.py --experiment ../experiments/2026-03-11_15-27-18_resnet18