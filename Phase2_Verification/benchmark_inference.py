"""
benchmark_inference.py
----------------------
Phase 2 benchmark using the actual ScanManager class (same as TestingPackageCode_02.py pattern).
Iterates over D_Fire dataset, calls sm.phase2() for each image, and collects:
  - Predictions + ground truth → confusion matrix, accuracy, precision, recall, F1
  - Per-stage timings from sm.last_phase2_timings → pipeline breakdown
  - Inference-only timing from predict_crops_majority_vote internal timing

Supports three backends via --backend flag:
  - pytorch:   Uses function_class_demo.ScanManager (PyTorch FP32)
  - tensorrt:  Uses function_class_demo_TensorRT.ScanManager (TensorRT FP16)
  - int8:      Uses function_class_demo_TensorRT.ScanManager (TensorRT INT8)

Uses the model weights already deployed inside wildfire_detector/ package:
  - wildfire_detector/best_model.pt          (for pytorch backend)
  - wildfire_detector/best_model_fp16.trt    (for tensorrt backend)
  - wildfire_detector/best_model_int8.trt    (for int8 backend)
  - wildfire_detector/config.yaml → phase2.model_name determines architecture

To test a different model: replace best_model.pt + update config.yaml model_name.

Covers thesis sections:
  - 5.3: Inference latency per model (inference_and_vote_ms column)
  - 5.5: Accuracy + timing comparison FP32 vs FP16 (confusion matrix + timings)
  - 5.6: Pipeline stage breakdown (metadata_and_region, crop_and_resize, batch_and_normalize, inference_and_vote)

Usage:
  	python Phase2_Verification/benchmark_inference.py --dataset Phase2_Verification/Benchmark_Test_Subset --backend pytorch --output_dir Phase2_Verification/benchmark_results/mobilenet_v2
	python Phase2_Verification/benchmark_inference.py --dataset Phase2_Verification/Benchmark_Test_Subset --backend tensorrt --output_dir Phase2_Verification/benchmark_results/mobilenet_v2
	python Phase2_Verification/benchmark_inference.py --dataset Phase2_Verification/Benchmark_Test_Subset --backend int8 --output_dir Phase2_Verification/benchmark_results/mobilenet_v2
"""

import os
import sys
import argparse
import copy
import time
import numpy as np
import torch
from torchvision import models
from PIL import Image
import yaml
import pandas as pd
import cv2
from tqdm import tqdm

sys.path.append(os.path.abspath(".."))

# Pipeline inference helpers (reused for the raw whole-image mode so raw uses the
# exact same model + normalization + inference/vote code as the crop pipeline).
from wildfire_detector.utils_phase2_flow import (
    predict_crops_majority_vote,
    predict_crops_majority_vote_RT,
)


# ===========================================================================
# YOLO label parsing
# ===========================================================================

def parse_yolo_label(label_path):
    """
    Parse YOLO label file (single merged bbox per image).
    Returns: (x_center, y_center, width, height) normalized, or None.
    """
    if not os.path.exists(label_path):
        return None
    with open(label_path, "r") as f:
        line = f.readline().strip()
    if not line:
        return None
    parts = line.split()
    return float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


# ===========================================================================
# Dataset loading
# ===========================================================================

def load_dataset(dataset_dir, fire_labels_subdir="labels"):
    """Load dataset: Fire/images + Fire/<fire_labels_subdir> + NoFire/images."""
    samples = []

    fire_images_dir = os.path.join(dataset_dir, "Fire", "images")
    fire_labels_dir = os.path.join(dataset_dir, "Fire", fire_labels_subdir)
    nofire_images_dir = os.path.join(dataset_dir, "NoFire")

    if os.path.isdir(fire_images_dir):
        for fname in sorted(os.listdir(fire_images_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                label_name = os.path.splitext(fname)[0] + ".txt"
                label_path = os.path.join(fire_labels_dir, label_name)
                samples.append({
                    "image_path": os.path.join(fire_images_dir, fname),
                    "image_name": fname,
                    "label": 1,
                    "label_path": label_path if os.path.exists(label_path) else None,
                })

    if os.path.isdir(nofire_images_dir):
        for fname in sorted(os.listdir(nofire_images_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append({
                    "image_path": os.path.join(nofire_images_dir, fname),
                    "image_name": fname,
                    "label": 0,
                    "label_path": None,
                })

    return samples


# ===========================================================================
# Helpers: prepare frame + metadata for sm.phase2() (TestingPackageCode_02 pattern)
# ===========================================================================

def prepare_frame(image_rgb, canvas_h, canvas_w):
    """
    Paste image onto center of (canvas_h x canvas_w) black canvas.
    Returns: (frame, scale, y_offset, x_offset) for bbox mapping.
    """
    H, W = image_rgb.shape[:2]
    scale = min(canvas_h / H, canvas_w / W)
    new_h = int(H * scale)
    new_w = int(W * scale)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    y_offset = (canvas_h - new_h) // 2
    x_offset = (canvas_w - new_w) // 2
    frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return frame, scale, y_offset, x_offset


def yolo_to_canvas_bbox(xc, yc, w, h, orig_H, orig_W, scale, y_offset, x_offset):
    """Convert YOLO normalized bbox → pixel coords on the canvas."""
    new_h = int(orig_H * scale)
    new_w = int(orig_W * scale)

    # Pixel coords in resized image
    c_min = int(round((xc - w / 2) * new_w))
    r_min = int(round((yc - h / 2) * new_h))
    c_max = int(round((xc + w / 2) * new_w))
    r_max = int(round((yc + h / 2) * new_h))

    # Shift to canvas coordinates
    r_min += y_offset
    r_max += y_offset
    c_min += x_offset
    c_max += x_offset

    return (r_min, c_min, r_max, c_max)


def compute_virtual_bbox(canvas_h, canvas_w, max_crop_factor):
    """Centered bbox such that the widest crop covers the full frame."""
    frame_min_dim = min(canvas_h, canvas_w)
    bbox_side = frame_min_dim / max_crop_factor
    center_r = canvas_h / 2.0
    center_c = canvas_w / 2.0
    half_side = bbox_side / 2.0
    r_min = int(round(center_r - half_side))
    c_min = int(round(center_c - half_side))
    r_max = int(round(center_r + half_side))
    c_max = int(round(center_c + half_side))
    return (r_min, c_min, r_max, c_max)


def build_metadata(sm, bbox):
    """Build phase2 metadata with bbox in pixel coords (like TestingPackageCode_02)."""
    md = copy.deepcopy(sm.dummy_md)
    # Format: [row_min, col_min, 0, row_max, col_max, 0]
    md["investigation_parameters"]["detected_bounding_box"] = [
        bbox[0], bbox[1], 0, bbox[2], bbox[3], 0
    ]
    return md


def classify_raw_frame(sm, frame_rgb, backend, image_size):
    """
    Classify the WHOLE frame (no bbox crop) by resizing the entire image to
    (image_size x image_size) and running it through the same model + normalization
    + inference/vote code used by the crop pipeline. This is the 'raw' crop mode:
    the full non-square frame is used directly (aspect ratio squished to 224x224),
    NOT a square crop taken over the image.

    Returns: (final_class:int, avg_conf:float, timings:dict[str,float] in seconds)
    """
    timings = {
        'input_and_metadata': 0.0,
        'region_validation': 0.0,
        'crop_extraction': 0.0,
    }
    t0 = time.perf_counter()

    # --- Resize whole frame (same interpolation as pipeline Stage 4) ---
    t_resize = time.perf_counter()
    t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()
    t = torch.nn.functional.interpolate(
        t.unsqueeze(0),
        size=(image_size, image_size),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)
    timings['resize_and_preprocess'] = time.perf_counter() - t_resize

    # --- Batch construction + normalization (same as pipeline Stage 5) ---
    t_batch = time.perf_counter()
    batch = t.unsqueeze(0)  # [1, 3, image_size, image_size]
    batch.div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    batch = (batch - mean) / std
    timings['batch_construction'] = time.perf_counter() - t_batch

    dummy_bbox = (0, 0, frame_rgb.shape[0], frame_rgb.shape[1])
    if backend == 'pytorch':
        final_class, avg_conf, infer_t = predict_crops_majority_vote(
            crops=batch.to(sm.device), model=sm.model, bbox=dummy_bbox,
            device=sm.device, verbose=False)
    else:
        final_class, avg_conf, infer_t = predict_crops_majority_vote_RT(
            crops=batch, model=sm.model, bbox=dummy_bbox, verbose=False)

    timings['model_inference'] = infer_t.get('inference', 0.0)
    timings['majority_vote'] = infer_t.get('postprocess', 0.0)
    timings['total'] = time.perf_counter() - t0
    return final_class, avg_conf, timings


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2 benchmark via ScanManager")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to D_Fire dataset (Fire/NoFire structure)")
    parser.add_argument("--backend", type=str, default="pytorch",
                        choices=["pytorch", "tensorrt", "int8"],
                        help="pytorch = FP32, tensorrt = FP16, int8 = INT8 (all via ScanManager)")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Output folder for results (default: benchmark_results)")
    parser.add_argument("--crop_factors", type=str, default=None,
                        help="Override crop factors (comma-separated, e.g. '1.4142' or '1.2247,1.4142,2.0')")
    parser.add_argument("--crop_mode", type=str, default="crops",
                        choices=["crops", "raw"],
                        help="crops = multi-scale square crops around the bbox (uses --crop_factors); "
                             "raw = classify the whole frame resized to net input (no cropping)")
    parser.add_argument("--fire_labels_subdir", type=str, default="labels",
                        help="Name of the Fire labels subfolder to use (default: 'labels'). "
                             "e.g. 'labels_biggest' to benchmark the biggest-fire bboxes.")
    args = parser.parse_args()

    # --- Create ScanManager (loads model from wildfire_detector/ package + warmup) ---
    if args.backend == "pytorch":
        from wildfire_detector.function_class_demo import ScanManager
        precision = "FP32"
    elif args.backend == "int8":
        from wildfire_detector.function_class_demo_TensorRT import ScanManager
        precision = "INT8"
    else:
        from wildfire_detector.function_class_demo_TensorRT import ScanManager
        precision = "FP16"

    # For INT8/FP16: create a temporary config file with patched precision
    # (avoids overwriting the real config.yaml which would destroy comments)
    _temp_config_path = None
    if args.backend in ("int8", "tensorrt"):
        import yaml as _yaml
        import tempfile

        # Locate a source config.yaml robustly. Prefer the installed package's
        # copy; fall back to the repo-local wildfire_detector/config.yaml sitting
        # one level up from this script. This avoids "config.yaml not found in
        # the venv" when the package resources are not laid out as expected.
        _config_path = None
        _candidates = []
        try:
            import importlib.resources as _pkg
            _candidates.append(str(_pkg.files("wildfire_detector") / "config.yaml"))
        except Exception:
            pass
        try:
            import wildfire_detector as _wd
            _candidates.append(os.path.join(os.path.dirname(_wd.__file__), "config.yaml"))
        except Exception:
            pass
        # Repo-local: <this_script>/../wildfire_detector/config.yaml
        _candidates.append(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "wildfire_detector", "config.yaml"))

        for _cand in _candidates:
            if _cand and os.path.exists(_cand):
                _config_path = _cand
                break
        if _config_path is None:
            raise FileNotFoundError(
                "Could not locate wildfire_detector/config.yaml. Looked in:\n  "
                + "\n  ".join(str(c) for c in _candidates)
            )

        print(f"[config] Using source config: {_config_path}")
        with open(_config_path, "r") as _f:
            _cfg = _yaml.safe_load(_f)
        _cfg["phase2"]["trt_precision"] = "int8" if args.backend == "int8" else "fp16"
        _temp_config_path = os.path.join(tempfile.gettempdir(), "wildfire_benchmark_config.yaml")
        with open(_temp_config_path, "w") as _f:
            _yaml.dump(_cfg, _f, default_flow_style=False)

    print(f"Initializing ScanManager ({args.backend})...")
    # Be robust to older installed wildfire_detector wheels that may not accept
    # the `verbose` and/or `config_path` kwargs.
    try:
        sm = ScanManager(config_path=_temp_config_path, verbose=False)
    except TypeError:
        try:
            sm = ScanManager(config_path=_temp_config_path)
        except TypeError:
            sm = ScanManager()

    # Read model name from config (whatever is currently set in wildfire_detector/config.yaml)
    model_name = sm.config["phase2"]["model_name"]

    # --- Config ---
    rgb_size = sm.config["image"]["rgb_size"]
    canvas_h, canvas_w = rgb_size[0], rgb_size[1]

    is_raw = args.crop_mode == "raw"
    image_size = sm.config["phase2"]["net_image_size"]

    # Override crop factors if specified via CLI
    if args.crop_factors:
        crop_factors = [float(x.strip()) for x in args.crop_factors.split(",")]
        sm.config["phase2"]["crop_factors"] = crop_factors
    else:
        crop_factors = sm.config["phase2"]["crop_factors"]
    max_crop_factor = max(crop_factors)

    print(f"\nModel: {model_name}")
    print(f"Backend: {args.backend} ({precision})")
    print(f"Canvas: {canvas_h}x{canvas_w}")
    if is_raw:
        print(f"Crop mode: raw (whole frame resized to {image_size}x{image_size}, no cropping)")
    else:
        print(f"Crop mode: crops | Crop factors: {crop_factors}")

    # Note: ScanManager.__init__ already performs warmup internally
    # (configured by config.yaml warmup.num_iterations)

    # --- Load dataset ---
    samples = load_dataset(args.dataset, args.fire_labels_subdir)
    fire_count = sum(1 for s in samples if s["label"] == 1)
    nofire_count = sum(1 for s in samples if s["label"] == 0)
    print(f"Dataset: {len(samples)} images (Fire: {fire_count}, NoFire: {nofire_count})")

    # --- Run inference ---
    print("\nRunning inference...")
    results = []
    skipped_no_bbox = 0

    for idx, sample in enumerate(tqdm(samples, desc="Inference", unit="img")):
        # Load image
        image_bgr = cv2.imread(sample["image_path"])
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H_orig, W_orig = image_rgb.shape[:2]

        # Determine bbox presence (label file exists AND is non-empty)
        parsed = None
        if sample["label"] == 1 and sample["label_path"] is not None:
            parsed = parse_yolo_label(sample["label_path"])
        has_bbox = parsed is not None

        # Fire images WITHOUT a bbox are skipped in EVERY mode (raw/single/multi)
        # so all strategies are evaluated on the exact same image set (equal
        # denominators for a fair comparison). NoFire images are always kept.
        if sample["label"] == 1 and not has_bbox:
            skipped_no_bbox += 1
            continue

        # Prepare frame on canvas (same as TestingPackageCode_02)
        frame, scale, y_off, x_off = prepare_frame(image_rgb, canvas_h, canvas_w)

        if is_raw:
            # Raw whole-frame classification: no bbox crop, use the full frame.
            final_class, avg_conf, timings = classify_raw_frame(
                sm, frame, args.backend, image_size)
            pred_label = 1 if final_class in ("Fire", 1) else 0
            confidence = avg_conf
        else:
            # Determine bbox for crop modes
            if sample["label"] == 1:
                xc, yc, w, h = parsed
                bbox = yolo_to_canvas_bbox(xc, yc, w, h, H_orig, W_orig, scale, y_off, x_off)
            else:
                bbox = compute_virtual_bbox(canvas_h, canvas_w, max_crop_factor)

            # Call phase2 through ScanManager (same as TestingPackageCode_02)
            md = build_metadata(sm, bbox)
            result = sm.phase2(frame, md)
            pred_label = 1 if result["final_prediction"] in ("Fire", 1) else 0
            confidence = result["confidence"]

            # Collect timings from instrumented _phase2_inner
            timings = sm.last_phase2_timings if hasattr(sm, 'last_phase2_timings') else {}

        results.append({
            "image": sample["image_name"],
            "true_label": sample["label"],
            "pred_label": pred_label,
            "confidence": round(confidence, 4),
            "input_and_metadata_ms": round(timings.get('input_and_metadata', 0) * 1000, 3),
            "region_validation_ms": round(timings.get('region_validation', 0) * 1000, 3),
            "crop_extraction_ms": round(timings.get('crop_extraction', 0) * 1000, 3),
            "resize_and_preprocess_ms": round(timings.get('resize_and_preprocess', 0) * 1000, 3),
            "batch_construction_ms": round(timings.get('batch_construction', 0) * 1000, 3),
            "model_inference_ms": round(timings.get('model_inference', 0) * 1000, 3),
            "majority_vote_ms": round(timings.get('majority_vote', 0) * 1000, 3),
            "total_ms": round(timings.get('total', 0) * 1000, 3),
        })

    if skipped_no_bbox > 0:
        print(f"\nSkipped {skipped_no_bbox} fire image(s) without a bbox label "
              f"(excluded from all crop modes for a fair comparison).")

    # --- Compute metrics ---
    df = pd.DataFrame(results)
    true_labels = df["true_label"].values
    pred_labels = df["pred_label"].values
    accuracy = (true_labels == pred_labels).mean()

    TP = ((pred_labels == 1) & (true_labels == 1)).sum()
    FP = ((pred_labels == 1) & (true_labels == 0)).sum()
    FN = ((pred_labels == 0) & (true_labels == 1)).sum()
    TN = ((pred_labels == 0) & (true_labels == 0)).sum()

    fire_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    fire_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fire_f1 = 2 * fire_precision * fire_recall / (fire_precision + fire_recall) if (fire_precision + fire_recall) > 0 else 0.0
    fpr_val = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr_val = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    # --- Print summary ---
    print("\n" + "=" * 60)
    print(f"PHASE 2 BENCHMARK — {model_name} ({precision})")
    print("=" * 60)
    print(f"  Accuracy:              {accuracy*100:.2f}%")
    print(f"  Fire Precision:        {fire_precision*100:.2f}%")
    print(f"  Fire Recall:           {fire_recall*100:.2f}%")
    print(f"  Fire F1:               {fire_f1*100:.2f}%")
    print(f"  FPR:                   {fpr_val*100:.2f}%")
    print(f"  FNR:                   {fnr_val*100:.2f}%")
    print(f"  ---")
    print(f"  Confusion Matrix:  TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    print(f"  ---")
    print(f"  Pipeline Timing (mean ms):")
    print(f"    input_and_metadata:    {df['input_and_metadata_ms'].mean():.3f}")
    print(f"    region_validation:     {df['region_validation_ms'].mean():.3f}")
    print(f"    crop_extraction:       {df['crop_extraction_ms'].mean():.3f}")
    print(f"    resize_and_preprocess: {df['resize_and_preprocess_ms'].mean():.3f}")
    print(f"    batch_construction:    {df['batch_construction_ms'].mean():.3f}")
    print(f"    model_inference:       {df['model_inference_ms'].mean():.3f}")
    print(f"    majority_vote:         {df['majority_vote_ms'].mean():.3f}")
    print(f"    total:                 {df['total_ms'].mean():.3f}")
    print(f"  ---")
    print(f"  Inference std:         {df['model_inference_ms'].std():.3f} ms")
    print(f"  FPS (total pipeline):  {1000.0 / df['total_ms'].mean():.1f}")
    print("=" * 60)

    # --- Save ---
    if is_raw:
        crops_suffix = "_raw"
    elif args.crop_factors:
        crops_suffix = f"_crops{len(crop_factors)}"
    else:
        crops_suffix = ""
    save_dir = os.path.join(args.output_dir, f"{model_name}_{args.backend}{crops_suffix}")
    os.makedirs(save_dir, exist_ok=True)

    df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    summary = {
        "model": model_name,
        "backend": args.backend,
        "precision": precision,
        "crop_mode": args.crop_mode,
        "num_crops": 1 if is_raw else len(crop_factors),
        "crop_factors": "raw" if is_raw else str(crop_factors),
        "dataset": args.dataset,
        "num_images": len(df),
        "accuracy": round(accuracy * 100, 2),
        "fire_precision": round(fire_precision * 100, 2),
        "fire_recall": round(fire_recall * 100, 2),
        "fire_f1": round(fire_f1 * 100, 2),
        "fpr": round(fpr_val * 100, 2),
        "fnr": round(fnr_val * 100, 2),
        "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
        "mean_input_and_metadata_ms": round(df['input_and_metadata_ms'].mean(), 3),
        "std_input_and_metadata_ms": round(df['input_and_metadata_ms'].std(), 3),
        "min_input_and_metadata_ms": round(df['input_and_metadata_ms'].min(), 3),
        "max_input_and_metadata_ms": round(df['input_and_metadata_ms'].max(), 3),
        "mean_region_validation_ms": round(df['region_validation_ms'].mean(), 3),
        "std_region_validation_ms": round(df['region_validation_ms'].std(), 3),
        "min_region_validation_ms": round(df['region_validation_ms'].min(), 3),
        "max_region_validation_ms": round(df['region_validation_ms'].max(), 3),
        "mean_crop_extraction_ms": round(df['crop_extraction_ms'].mean(), 3),
        "std_crop_extraction_ms": round(df['crop_extraction_ms'].std(), 3),
        "min_crop_extraction_ms": round(df['crop_extraction_ms'].min(), 3),
        "max_crop_extraction_ms": round(df['crop_extraction_ms'].max(), 3),
        "mean_resize_and_preprocess_ms": round(df['resize_and_preprocess_ms'].mean(), 3),
        "std_resize_and_preprocess_ms": round(df['resize_and_preprocess_ms'].std(), 3),
        "min_resize_and_preprocess_ms": round(df['resize_and_preprocess_ms'].min(), 3),
        "max_resize_and_preprocess_ms": round(df['resize_and_preprocess_ms'].max(), 3),
        "mean_batch_construction_ms": round(df['batch_construction_ms'].mean(), 3),
        "std_batch_construction_ms": round(df['batch_construction_ms'].std(), 3),
        "min_batch_construction_ms": round(df['batch_construction_ms'].min(), 3),
        "max_batch_construction_ms": round(df['batch_construction_ms'].max(), 3),
        "mean_model_inference_ms": round(df['model_inference_ms'].mean(), 3),
        "std_model_inference_ms": round(df['model_inference_ms'].std(), 3),
        "min_model_inference_ms": round(df['model_inference_ms'].min(), 3),
        "max_model_inference_ms": round(df['model_inference_ms'].max(), 3),
        "mean_majority_vote_ms": round(df['majority_vote_ms'].mean(), 3),
        "std_majority_vote_ms": round(df['majority_vote_ms'].std(), 3),
        "min_majority_vote_ms": round(df['majority_vote_ms'].min(), 3),
        "max_majority_vote_ms": round(df['majority_vote_ms'].max(), 3),
        "mean_total_ms": round(df['total_ms'].mean(), 3),
        "std_total_ms": round(df['total_ms'].std(), 3),
        "min_total_ms": round(df['total_ms'].min(), 3),
        "max_total_ms": round(df['total_ms'].max(), 3),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "summary.csv"), index=False)

    print(f"\nResults saved to: {save_dir}/")


if __name__ == "__main__":
    main()
