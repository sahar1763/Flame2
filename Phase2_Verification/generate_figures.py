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


# Optional filters applied to every loaded summary (set from CLI in main()).
# Needed when a results tree contains several crop strategies per model+backend
# (e.g. the IAI crop_results tree: raw / crops1 / crops3), so a single strategy
# can be isolated for the single-crop figures. None = no filtering.
_FILTER_CROP_MODE = None   # e.g. "crops" or "raw"
_FILTER_NUM_CROPS = None   # e.g. 1 or 3


def load_all_summaries(results_dir):
    """Load all summary.csv files found anywhere under results_dir (recursive).

    Supports both a flat layout (results_dir/<model>_<backend>/summary.csv) and a
    nested layout (results_dir/<model>/<model>_<backend>/summary.csv).

    If the module-level _FILTER_CROP_MODE / _FILTER_NUM_CROPS are set, rows are
    restricted to that crop strategy (columns crop_mode / num_crops).
    """
    summaries = []
    paths = []
    for root, _dirs, files in os.walk(results_dir):
        if "summary.csv" in files:
            paths.append(os.path.join(root, "summary.csv"))
    for summary_path in sorted(paths):
        df = pd.read_csv(summary_path)
        df["subfolder"] = os.path.basename(os.path.dirname(summary_path))
        summaries.append(df)
    if not summaries:
        return pd.DataFrame()
    combined = pd.concat(summaries, ignore_index=True)
    if _FILTER_CROP_MODE is not None and "crop_mode" in combined.columns:
        combined = combined[combined["crop_mode"] == _FILTER_CROP_MODE]
    if _FILTER_NUM_CROPS is not None and "num_crops" in combined.columns:
        combined = combined[combined["num_crops"].astype(int) == _FILTER_NUM_CROPS]
    return combined.reset_index(drop=True)


def load_predictions(results_dir, subfolder):
    """Load predictions.csv for a given subfolder (searched recursively)."""
    for root, _dirs, files in os.walk(results_dir):
        if os.path.basename(root) == subfolder and "predictions.csv" in files:
            return pd.read_csv(os.path.join(root, "predictions.csv"))
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

# Trainable parameter counts are COMPUTED from the actual model architecture
# (with the same 2-class head the benchmark rebuilds), never hard-coded.
_PARAM_CACHE = {}


def _build_model_with_head(model_name):
    """Rebuild the torchvision architecture with a 2-class head (same logic as
    classifier_latency_benchmark.TorchClassifier / ScanManager)."""
    import torch
    from torchvision import models
    model = getattr(models, model_name)(weights=None)
    num_classes = 2
    name = model_name.lower()
    if "resnet" in name:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif "vgg" in name or "alexnet" in name:
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif "mobilenet" in name:
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def get_params_m(model_name):
    """Return trainable parameter count in millions, computed from the actual
    architecture. Returns None if torch/torchvision is unavailable or the model
    name is unknown (callers then skip the parameter annotation, no fallback)."""
    if model_name in _PARAM_CACHE:
        return _PARAM_CACHE[model_name]
    try:
        model = _build_model_with_head(model_name)
        params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    except Exception as exc:  # torch missing or unknown arch
        print(f"  WARN: could not compute params for {model_name}: {exc}")
        params_m = None
    _PARAM_CACHE[model_name] = params_m
    return params_m


def find_checkpoint(weights_dir, model_name):
    """Locate <weights_dir>/**/<*_model_name>/checkpoints/best_model.pt.

    Matches an experiment folder whose name ends with '_<model_name>'. Returns the
    .pt path (newest by folder name if several) or None. Nothing is guessed — if no
    matching file exists, size columns are simply left blank.
    """
    matches = []
    for root, _dirs, files in os.walk(weights_dir):
        if "best_model.pt" in files:
            parent = os.path.basename(os.path.dirname(root))  # experiment folder
            if parent.endswith("_" + model_name) or parent == model_name:
                matches.append(os.path.join(root, "best_model.pt"))
    if not matches:
        return None
    matches.sort()
    return matches[-1]


def get_weights_size_mb(model_name, weights_dir):
    """Read the REAL model-weights size from the actual checkpoint (no assumptions).

    Re-serializes ONLY model_state (the network weights), excluding optimizer state
    and other training bookkeeping, and returns its size in MB. Returns None if the
    file or weights cannot be read.
    """
    path = find_checkpoint(weights_dir, model_name)
    if path is None:
        print(f"  WARN: no best_model.pt found for {model_name} under {weights_dir}")
        return None
    try:
        import io
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
        buf = io.BytesIO()
        torch.save(state, buf)
        return buf.getbuffer().nbytes / (1024 * 1024)
    except Exception as exc:
        print(f"  WARN: could not read weights size for {model_name}: {exc}")
        return None


def get_artifact_sizes_mb(model_name, weights_dir):
    """Read REAL on-disk sizes (MB) of the deployment artifacts that sit next to
    the checkpoint: best_model.onnx, best_model_fp16.trt, best_model_int8.trt.

    Returns a dict {onnx_mb, trt_fp16_mb, trt_int8_mb}; any missing file -> None.
    Sizes are read directly from the files, nothing is assumed.
    """
    result = {"onnx_mb": None, "trt_fp16_mb": None, "trt_int8_mb": None}
    ckpt_path = find_checkpoint(weights_dir, model_name)
    if ckpt_path is None:
        return result
    ckpt_dir = os.path.dirname(ckpt_path)
    artifacts = {
        "onnx_mb": "best_model.onnx",
        "trt_fp16_mb": "best_model_fp16.trt",
        "trt_int8_mb": "best_model_int8.trt",
    }
    for key, fname in artifacts.items():
        fpath = os.path.join(ckpt_dir, fname)
        if os.path.isfile(fpath):
            result[key] = os.path.getsize(fpath) / (1024 * 1024)
    return result


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

    # Add parameter count (computed from the architecture)
    df["params_m"] = df["model"].map(get_params_m)
    if df["params_m"].isna().any():
        missing = df[df["params_m"].isna()]["model"].tolist()
        print(f"  Skipping param labels for (uncomputable): {missing}")
    df = df.sort_values("params_m")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar chart: x = model (sorted by params), y = inference time
    bars = ax.bar(range(len(df)), df["mean_model_inference_ms"],
                  color="steelblue", edgecolor="black", width=0.6)

    # Label each bar with model name + params (omit param note if uncomputable)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([
        (f"{row['model']}\n({row['params_m']:.1f}M)"
         if pd.notna(row["params_m"]) else f"{row['model']}")
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

    # Bubble size proportional to parameter count (computed from architecture)
    df["params_m"] = df["model"].map(get_params_m)
    if df["params_m"].notna().any():
        pmax = df["params_m"].max()
        sizes = df["params_m"].fillna(pmax * 0.1) / pmax * 400 + 80
    else:
        sizes = 200  # uniform size if no counts available

    scatter = ax.scatter(df["mean_model_inference_ms"], df["fire_f1"],
                         s=sizes, c="steelblue", edgecolors="black",
                         alpha=0.7, zorder=5)

    # Label each point with model name + param count (omit note if uncomputable)
    for _, row in df.iterrows():
        label = (f"{row['model']}\n({row['params_m']:.1f}M)"
                 if pd.notna(row["params_m"]) else f"{row['model']}")
        ax.annotate(label,
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
# Deployment format comparison (FP32 / FP16 / INT8 latency bars)
# ===========================================================================

def generate_deployment_format_comparison(results_dir, output_dir):
    """Bar chart of mean inference time per deployment format for the same model.

    Each bar is annotated with the measured latency, the speedup relative to the
    FP32 baseline, and the fire F1-score. All values are read from summary.csv;
    nothing is hard-coded. Saved per model folder next to precision_comparison.png.
    """
    summaries = load_all_summaries(results_dir)
    if summaries.empty:
        print("No summaries found.")
        return

    # Find models that have multiple backends
    model_counts = summaries.groupby("model")["backend"].nunique()
    multi_backend_models = model_counts[model_counts > 1].index.tolist()

    if not multi_backend_models:
        print("No model has results from multiple backends — need at least FP32 + FP16.")
        return

    precision_order = {"FP32": 0, "FP16": 1, "INT8": 2}
    backend_display = {"pytorch": "PyTorch", "tensorrt": "TensorRT", "int8": "TensorRT"}
    colors = {"FP32": "#4C72B0", "FP16": "#55A868", "INT8": "#C44E52"}

    for model_name in multi_backend_models:
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        df = summaries[summaries["model"] == model_name].copy()
        df["sort_key"] = df["precision"].map(precision_order).fillna(99)
        df = df.sort_values("sort_key")

        # Speedup relative to the FP32 baseline (falls back to slowest if no FP32)
        baseline_rows = df[df["precision"] == "FP32"]["mean_model_inference_ms"]
        baseline = baseline_rows.iloc[0] if not baseline_rows.empty else df["mean_model_inference_ms"].max()

        labels = [f"{backend_display.get(b, b)}\n{p}"
                  for b, p in zip(df["backend"], df["precision"])]
        latencies = df["mean_model_inference_ms"].values
        f1_vals = df["fire_f1"].values
        bar_colors = [colors.get(p, "gray") for p in df["precision"]]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(range(len(df)), latencies, color=bar_colors,
                      edgecolor="black", width=0.6, zorder=3)

        for bar, lat, f1 in zip(bars, latencies, f1_vals):
            speedup = baseline / lat if lat > 0 else 0.0
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + latencies.max() * 0.01,
                    f"{lat:.2f} ms\n{speedup:.1f}\u00d7 | F1 {f1:.1f}%",
                    ha="center", va="bottom", fontsize=10)

        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Mean Inference Time [ms]", fontsize=12)
        ax.set_title(f"{model_name} \u2014 Deployment Format Comparison", fontsize=13)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_ylim(0, latencies.max() * 1.2)
        plt.tight_layout()

        save_path = os.path.join(model_dir, "deployment_format_comparison.png")
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

    # Skip if this results set lacks per-stage timing columns (e.g. the
    # single-image classifier benchmark, which only records model_inference).
    if not all(stage in summaries.columns for stage in stages):
        print("  No pipeline stage columns found — skipping pipeline breakdown.")
        return

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

def export_combined_table(results_dir, output_dir, weights_dir=None):
    """Export combined CSV with all models × backends for easy thesis table creation.

    Adds real, computed columns: params_m (from architecture) and, when
    weights_dir is given, weights_mb read from the actual best_model.pt files
    (model weights only, optimizer state excluded). No values are hard-coded.
    """
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
    table = summaries[available_cols].copy()

    # Real computed parameter count (from architecture, not hard-coded)
    table["params_m"] = table["model"].map(lambda m: round(get_params_m(m), 3)
                                           if get_params_m(m) is not None else None)

    # Real weights size read from the actual checkpoint files (optimizer excluded)
    size_cache = {}
    artifact_cache = {}
    if weights_dir:
        for m in table["model"].unique():
            size_cache[m] = get_weights_size_mb(m, weights_dir)
            artifact_cache[m] = get_artifact_sizes_mb(m, weights_dir)
        table["weights_mb"] = table["model"].map(
            lambda m: round(size_cache[m], 2) if size_cache[m] is not None else None)
        for col in ("onnx_mb", "trt_fp16_mb", "trt_int8_mb"):
            table[col] = table["model"].map(
                lambda m, c=col: round(artifact_cache[m][c], 2)
                if artifact_cache[m][c] is not None else None)

    table = table.sort_values(["model", "backend"])
    save_path = os.path.join(output_dir, "combined_results_table.csv")
    table.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")

    # Dedicated per-model complexity table (one row per model, from real sources)
    model_rows = []
    for m in sorted(table["model"].unique()):
        row = {"model": m, "params_m": round(get_params_m(m), 3)
               if get_params_m(m) is not None else None}
        if weights_dir:
            row["weights_mb"] = (round(size_cache[m], 2)
                                 if size_cache[m] is not None else None)
            for col in ("onnx_mb", "trt_fp16_mb", "trt_int8_mb"):
                val = artifact_cache[m][col]
                row[col] = round(val, 2) if val is not None else None
        model_rows.append(row)
    complexity = pd.DataFrame(model_rows)
    comp_path = os.path.join(output_dir, "model_complexity_table.csv")
    complexity.to_csv(comp_path, index=False)
    print(f"  Saved: {comp_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate thesis Chapter 5 figures from benchmark results")
    parser.add_argument("--results_dir", type=str, default="benchmark_results",
                        help="Path to benchmark results folder")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Output folder for generated figures")
    parser.add_argument("--weights_dir", type=str, default=None,
                        help="Folder with <*_model>/checkpoints/best_model.pt files. "
                             "When given, real weights sizes are read into the tables.")
    parser.add_argument("--crop_mode", type=str, default=None,
                        choices=["crops", "raw"],
                        help="Restrict to one crop mode (needed when the results tree "
                             "has several strategies per model, e.g. the IAI tree).")
    parser.add_argument("--num_crops", type=int, default=None,
                        help="Restrict to runs with this crop count (1 = single \u221a2 "
                             "crop, 3 = multi-crop). Combine with --crop_mode crops.")
    args = parser.parse_args()

    global _FILTER_CROP_MODE, _FILTER_NUM_CROPS
    _FILTER_CROP_MODE = args.crop_mode
    _FILTER_NUM_CROPS = args.num_crops

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

    # Fig 12b: Deployment format comparison (latency bars + speedup + F1)
    print("\n[Fig 12b] Deployment format comparison:")
    generate_deployment_format_comparison(args.results_dir, args.output_dir)

    # Pipeline breakdown (Section 5.6)
    print("\n[Pipeline] Stage breakdown:")
    generate_pipeline_breakdown(args.results_dir, args.output_dir)

    # Combined table export
    print("\n[Table] Combined results:")
    export_combined_table(args.results_dir, args.output_dir, args.weights_dir)

    print("\n" + "=" * 50)
    print("Done! All figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
