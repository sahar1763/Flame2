"""
generate_figures.py
-------------------
Reads benchmark results CSVs and generates thesis Chapter 5 figures.

Expected input: benchmark_results/ folder containing subfolders like:
    resnet18_pytorch/predictions.csv + summary.csv
    resnet18_tensorrt/predictions.csv + summary.csv
    vgg16_pytorch/predictions.csv + summary.csv
    ...

Generates:
  - Fig 9:  Confusion matrices per model (from predictions.csv)
  - Fig 10: Inference time bar chart across models
  - Fig 11: F1-score vs latency scatter (accuracy-latency trade-off)
  - Fig 12: FP32 vs FP16 vs INT8 latency comparison (selected model)
  - Table data exports for thesis tables

Usage:
  python generate_figures.py --results_dir benchmark_results --output_dir figures
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures


def load_all_summaries(results_dir):
    """Load all summary.csv files from subfolders."""
    summaries = []
    for subfolder in sorted(os.listdir(results_dir)):
        summary_path = os.path.join(results_dir, subfolder, "summary.csv")
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df["subfolder"] = subfolder
            summaries.append(df)
    if summaries:
        return pd.concat(summaries, ignore_index=True)
    return pd.DataFrame()


def load_predictions(results_dir, subfolder):
    """Load predictions.csv from a specific subfolder."""
    path = os.path.join(results_dir, subfolder, "predictions.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ===========================================================================
# Figure 9: Confusion matrices per model
# ===========================================================================

def generate_confusion_matrices(results_dir, output_dir, backend="pytorch"):
    """Generate individual confusion matrix plot per model with given backend."""
    summaries = load_all_summaries(results_dir)
    if summaries.empty:
        print("No summaries found for confusion matrices.")
        return

    # Filter by backend
    df = summaries[summaries["backend"] == backend].copy()
    if df.empty:
        print(f"No results found for backend={backend}")
        return

    for _, row in df.iterrows():
        model_dir = os.path.join(output_dir, row["model"])
        os.makedirs(model_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        cm = np.array([[row["TN"], row["FP"]],
                       [row["FN"], row["TP"]]])

        ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{row['model']} ({row['precision']})", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["NoFire", "Fire"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["NoFire", "Fire"])

        # Annotate cells
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, str(int(val)), ha="center", va="center",
                        fontsize=14, fontweight="bold", color=color)

        plt.tight_layout()
        save_path = os.path.join(model_dir, f"confusion_matrix_{backend}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")


# ===========================================================================
# Figure 10: Model complexity vs inference time
# ===========================================================================

# Approximate parameter counts (millions) for supported models (trainable only)
MODEL_PARAMS_M = {
    "alexnet": 57.0,
    "vgg16": 134.3,
    "resnet18": 11.2,
    "resnet34": 21.3,
    "resnet50": 23.5,
    "mobilenet_v2": 2.2,
}


def generate_inference_bar_chart(results_dir, output_dir, backend="pytorch"):
    """Scatter/bar: model parameter count vs inference time."""
    summaries = load_all_summaries(results_dir)
    df = summaries[summaries["backend"] == backend].copy()

    if df.empty:
        print(f"No results for backend={backend}")
        return

    if len(df) < 2:
        print(f"Only 1 model found for backend={backend} — skipping (need 2+ for comparison).")
        return

    # Add parameter count
    df["params_m"] = df["model"].map(MODEL_PARAMS_M)
    df = df.sort_values("params_m")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar chart: x = model (sorted by params), y = inference time
    bars = ax.bar(range(len(df)), df["mean_model_inference_ms"],
                  color="steelblue", edgecolor="black", width=0.6)

    # Label each bar with model name + ~params
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{row['model']}\n(~{row['params_m']:.1f}M)" 
                        for _, row in df.iterrows()], fontsize=9)

    # Add time value above each bar
    for bar, val in zip(bars, df["mean_model_inference_ms"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.2f} ms", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Inference Time [ms]", fontsize=12)
    ax.set_title(f"Model Complexity vs. Inference Latency ({backend.upper()})", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    # Add headroom so labels don't clip
    ax.set_ylim(0, df["mean_model_inference_ms"].max() * 1.2)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"fig10_inference_time_{backend}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ===========================================================================
# Figure 11: F1-score vs latency scatter (accuracy-latency trade-off)
# ===========================================================================

def generate_tradeoff_scatter(results_dir, output_dir, backend="pytorch"):
    """Scatter plot: x=inference time, y=F1-score — cross-model comparison only."""
    summaries = load_all_summaries(results_dir)
    df = summaries[summaries["backend"] == backend].copy()

    if df.empty:
        print(f"No results for backend={backend}")
        return

    if len(df) < 2:
        print(f"Only 1 model found for backend={backend} — skipping (need 2+ for comparison).")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    # Bubble size proportional to parameter count
    df["params_m"] = df["model"].map(MODEL_PARAMS_M)
    sizes = df["params_m"] / df["params_m"].max() * 400 + 80  # scale to visible range

    scatter = ax.scatter(df["mean_model_inference_ms"], df["fire_f1"],
                         s=sizes, c="steelblue", edgecolors="black",
                         alpha=0.7, zorder=5)

    # Label each point with model name + param count
    for _, row in df.iterrows():
        ax.annotate(f"{row['model']}\n(~{row['params_m']:.1f}M)",
                    (row["mean_model_inference_ms"], row["fire_f1"]),
                    textcoords="offset points", xytext=(10, -5), fontsize=9)

    ax.set_xlabel("Mean Inference Time [ms]", fontsize=12)
    ax.set_ylabel("Fire F1-Score [%]", fontsize=12)
    ax.set_title("Accuracy–Latency Trade-Off", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Add slight headroom for labels
    y_min = df["fire_f1"].min()
    y_max = df["fire_f1"].max()
    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    x_max = df["mean_model_inference_ms"].max()
    ax.set_xlim(0, x_max * 1.15)

    # Add note about bubble size
    ax.text(0.98, 0.02, "Bubble size ∝ parameter count",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            fontstyle="italic", color="gray")

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"fig11_tradeoff_{backend}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ===========================================================================
# Figure 12: FP32 vs FP16 vs INT8 comparison (selected model)
# ===========================================================================

def generate_precision_comparison(results_dir, output_dir):
    """Bar chart comparing FP32, FP16, INT8 for the same model — saved per model folder."""
    summaries = load_all_summaries(results_dir)
    if summaries.empty:
        print("No summaries found.")
        return

    # Find model that has multiple backends
    model_counts = summaries.groupby("model")["backend"].nunique()
    multi_backend_models = model_counts[model_counts > 1].index.tolist()

    if not multi_backend_models:
        print("No model has results from multiple backends — need at least FP32 + FP16.")
        return

    for model_name in multi_backend_models:
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        df = summaries[summaries["model"] == model_name].copy()

        # Custom sort
        precision_order = {"FP32": 0, "FP16": 1, "INT8": 2}
        df["sort_key"] = df["precision"].map(precision_order)
        df = df.sort_values("sort_key")

        colors = {"FP32": "#4C72B0", "FP16": "#55A868", "INT8": "#C44E52"}

        fig, ax = plt.subplots(figsize=(7, 5))

        x_vals = df["mean_model_inference_ms"].values
        y_vals = df["fire_f1"].values
        bar_colors = [colors.get(p, "gray") for p in df["precision"]]

        bars = ax.bar(x_vals, y_vals, width=x_vals.max() * 0.08,
                      color=bar_colors, edgecolor="black", zorder=5)

        # Add legend entries
        for bar, prec in zip(bars, df["precision"]):
            bar.set_label(prec)

        ax.set_xlabel("Mean Inference Time [ms]", fontsize=12)
        ax.set_ylabel("Fire F1-Score [%]", fontsize=12)
        ax.set_title(f"{model_name} — Precision Comparison", fontsize=12)
        ax.legend(fontsize=10, title="Precision")
        ax.grid(True, alpha=0.3)

        # Headroom
        x_max = df["mean_model_inference_ms"].max()
        ax.set_xlim(0, x_max * 1.15)
        y_min = df["fire_f1"].min()
        y_max = df["fire_f1"].max()
        y_margin = (y_max - y_min) * 0.15 if y_max > y_min else 2
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        plt.tight_layout()
        save_path = os.path.join(model_dir, "precision_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")


# ===========================================================================
# Pipeline breakdown stacked bar (Section 5.6)
# ===========================================================================

def generate_pipeline_breakdown(results_dir, output_dir):
    """Per-model pipeline stage breakdown — all backends on same figure per model."""
    summaries = load_all_summaries(results_dir)
    if summaries.empty:
        return

    backend_labels = {"pytorch": "FP32", "tensorrt": "FP16", "int8": "INT8"}
    backend_order = ["pytorch", "tensorrt", "int8"]

    stages = ["mean_input_and_metadata_ms", "mean_region_validation_ms",
              "mean_crop_extraction_ms", "mean_resize_and_preprocess_ms",
              "mean_batch_construction_ms", "mean_model_inference_ms",
              "mean_majority_vote_ms"]
    labels = ["Input & Metadata", "Region Validation", "Crop Extraction",
              "Resize & Preprocess", "Batch Construction", "Model Inference",
              "Majority Vote"]
    colors = ["#FDB462", "#FCCDE5", "#80B1D3", "#8DD3C7",
              "#B3DE69", "#FB8072", "#BEBADA"]

    # Per-model figure: all backends stacked on same chart
    for model_name in summaries["model"].unique():
        model_df = summaries[summaries["model"] == model_name].copy()
        # Sort by backend order
        model_df["sort_key"] = model_df["backend"].map({b: i for i, b in enumerate(backend_order)})
        model_df = model_df.sort_values("sort_key")

        if model_df.empty:
            continue

        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        n_bars = len(model_df)
        fig, ax = plt.subplots(figsize=(9, 2 + n_bars * 1.2))

        y_labels = [backend_labels.get(row["backend"], row["backend"]) for _, row in model_df.iterrows()]

        for idx, (_, row) in enumerate(model_df.iterrows()):
            bottom = 0
            for stage, label, color in zip(stages, labels, colors):
                val = row[stage]
                bar = ax.barh([idx], [val], left=[bottom], label=label if idx == 0 else "",
                              color=color, edgecolor="white", height=0.6)
                bottom += val
            # Total time label at end of bar
            ax.text(bottom + 0.2, idx, f"{bottom:.1f} ms", va="center", fontsize=10)

        ax.set_yticks(range(n_bars))
        ax.set_yticklabels(y_labels, fontsize=11)
        ax.set_xlabel("Time [ms]", fontsize=12)
        ax.set_title(f"Pipeline Breakdown — {model_name}", fontsize=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(model_dir, "pipeline_breakdown.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # Combined comparison at top level (one backend — prefer tensorrt)
    for backend in backend_order:
        df = summaries[summaries["backend"] == backend].copy()
        if len(df) > 1:
            df = df.sort_values("mean_total_ms")
            backend_label = backend_labels.get(backend, backend)
            fig, ax = plt.subplots(figsize=(10, 5))
            bottoms = np.zeros(len(df))
            for stage, label, color in zip(stages, labels, colors):
                values = df[stage].values
                ax.barh(df["model"], values, left=bottoms, label=label, color=color, edgecolor="white")
                bottoms += values
            ax.set_xlabel("Time [ms]", fontsize=12)
            ax.set_title(f"Pipeline Stage Breakdown per Model ({backend_label})", fontsize=13)
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()

            save_path = os.path.join(output_dir, f"fig_pipeline_breakdown_{backend}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {save_path}")


# ===========================================================================
# Export combined table (for thesis)
# ===========================================================================

def export_combined_table(results_dir, output_dir):
    """Export combined CSV with all models × backends for easy thesis table creation."""
    summaries = load_all_summaries(results_dir)
    if summaries.empty:
        return

    cols = ["model", "backend", "precision", "accuracy", "fire_precision",
            "fire_recall", "fire_f1", "fpr", "fnr",
            "mean_input_and_metadata_ms", "std_input_and_metadata_ms", "min_input_and_metadata_ms", "max_input_and_metadata_ms",
            "mean_region_validation_ms", "std_region_validation_ms", "min_region_validation_ms", "max_region_validation_ms",
            "mean_crop_extraction_ms", "std_crop_extraction_ms", "min_crop_extraction_ms", "max_crop_extraction_ms",
            "mean_resize_and_preprocess_ms", "std_resize_and_preprocess_ms", "min_resize_and_preprocess_ms", "max_resize_and_preprocess_ms",
            "mean_batch_construction_ms", "std_batch_construction_ms", "min_batch_construction_ms", "max_batch_construction_ms",
            "mean_model_inference_ms", "std_model_inference_ms", "min_model_inference_ms", "max_model_inference_ms",
            "mean_majority_vote_ms", "std_majority_vote_ms", "min_majority_vote_ms", "max_majority_vote_ms",
            "mean_total_ms", "std_total_ms", "min_total_ms", "max_total_ms"]
    available_cols = [c for c in cols if c in summaries.columns]
    table = summaries[available_cols].sort_values(["model", "backend"])

    save_path = os.path.join(output_dir, "combined_results_table.csv")
    table.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate thesis Chapter 5 figures from benchmark results")
    parser.add_argument("--results_dir", type=str, default="benchmark_results",
                        help="Path to benchmark results folder")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Output folder for generated figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating thesis figures...")
    print("=" * 50)

    # Fig 9: Confusion matrices (all available backends)
    print("\n[Fig 9] Confusion matrices:")
    for backend in ["pytorch", "tensorrt", "int8"]:
        generate_confusion_matrices(args.results_dir, args.output_dir, backend=backend)

    # Fig 10: Inference time bar chart (all available backends)
    print("\n[Fig 10] Inference time bar chart:")
    for backend in ["pytorch", "tensorrt", "int8"]:
        generate_inference_bar_chart(args.results_dir, args.output_dir, backend=backend)

    # Fig 11: Accuracy-latency trade-off (all available backends)
    print("\n[Fig 11] Accuracy-latency trade-off:")
    for backend in ["pytorch", "tensorrt", "int8"]:
        generate_tradeoff_scatter(args.results_dir, args.output_dir, backend=backend)

    # Fig 12: FP32 vs FP16 vs INT8
    print("\n[Fig 12] Precision comparison (FP32/FP16/INT8):")
    generate_precision_comparison(args.results_dir, args.output_dir)

    # Pipeline breakdown (Section 5.6)
    print("\n[Pipeline] Stage breakdown:")
    generate_pipeline_breakdown(args.results_dir, args.output_dir)

    # Combined table export
    print("\n[Table] Combined results:")
    export_combined_table(args.results_dir, args.output_dir)

    print("\n" + "=" * 50)
    print("Done! All figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
