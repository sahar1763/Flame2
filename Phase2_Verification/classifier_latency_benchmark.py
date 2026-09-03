"""
classifier_latency_benchmark.py
-------------------------------
Plain single-image classifier benchmark for Chapter 5 (Sections 5.3 and 5.5).

This measures the model EXACTLY the way it was trained and tested:
    whole image  ->  resize 224x224  ->  ImageNet normalize  ->  ONE forward pass.

There is NO bounding box, NO multi-scale cropping and NO majority voting here.
(That full verification pipeline is measured separately by benchmark_inference.py
for Section 5.6.) Keeping this path identical to training guarantees the accuracy
reported here matches the server / W&B numbers, so there is no "85% on the server
but 95% on the Orin" inconsistency.

Supported precisions (one run each):
    --precision fp32   -> PyTorch .pt checkpoint (backend label: pytorch)
    --precision fp16   -> TensorRT .trt engine   (backend label: tensorrt)
    --precision int8   -> TensorRT .trt engine   (backend label: int8)

Supported architectures: anything in torchvision that the project trains, e.g.
    resnet18, resnet34, resnet50, alexnet, vgg16, mobilenet_v2
(the head is rebuilt for 2 classes exactly like function_class_demo.ScanManager).

Output (same folder layout + summary.csv schema as benchmark_inference.py, so
generate_figures.py consumes it with NO changes):
    <output_dir>/<model>_<backend>/
        predictions.csv   per-image: image, true_label, pred_label, confidence, model_inference_ms
        summary.csv        one row: model, backend, precision, accuracy, fire_*,
                           TP/FP/FN/TN, mean/std/min/max/p95 model_inference_ms

Two dataset input modes (pick one):

  1) Folder mode (--dataset): same Fire/NoFire structure as the rest of Phase 2
         <dataset>/Fire/images/   -> label 1
         <dataset>/NoFire/        -> label 0

  2) labels.csv mode (--images_dir + --labels_csv): SAME as training / test.py
         <images_dir>/            -> flat images referenced by id
         labels.csv columns: id, dataset, fire   (fire==1 -> label 1, else 0)
     Optionally restrict to specific source datasets with --test_datasets
     (e.g. the held-out FireSmoke sets), mirroring test_only in training.

Usage examples (run from the Phase2_Verification/ folder):
  # FP32 (PyTorch), labels.csv mode -- matches training exactly (recommended)
  python Phase2_Verification/classifier_latency_benchmark.py --model_name mobilenet_v2 --precision fp32 \
      --weights wildfire_detector/best_model.pt \
      --images_dir TestSet --labels_csv TestSet/labels.csv \
      --test_datasets FireSmokeDataset FireSmokeNEWdataset --output_dir Phase2_Verification/inference_results/mobilenet_v2

  # FP16 (TensorRT engine, built on the Orin)
  python Phase2_Verification/classifier_latency_benchmark.py --model_name mobilenet_v2 --precision fp16 \
      --weights wildfire_detector/best_model_fp16.trt \
      --images_dir TestSet --labels_csv TestSet/labels.csv \
      --test_datasets FireSmokeDataset FireSmokeNEWdataset --output_dir Phase2_Verification/inference_results/mobilenet_v2

  # INT8 (TensorRT engine, built on the Orin)
  python Phase2_Verification/classifier_latency_benchmark.py --model_name mobilenet_v2 --precision int8 \
      --weights wildfire_detector/best_model_int8.trt \
      --images_dir TestSet --labels_csv TestSet/labels.csv \
      --test_datasets FireSmokeDataset FireSmokeNEWdataset --output_dir Phase2_Verification/inference_results/mobilenet_v2

  # (alternative) folder mode with a Fire/NoFire directory
  python classifier_latency_benchmark.py --model_name resnet18 --precision fp32 \
      --weights wildfire_detector/best_model.pt --dataset TestSet

Then generate the figures/tables (Fig 9/10/11/12 + combined table):
  python generate_figures.py --results_dir benchmark_results --output_dir figures
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(".."))

# ImageNet normalization — identical to training / function_class_demo.py
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

# fp32 -> pytorch, fp16 -> tensorrt, int8 -> int8
# (backend label matches what generate_figures.py filters on)
PRECISION_TO_BACKEND = {"fp32": "pytorch", "fp16": "tensorrt", "int8": "int8"}
PRECISION_LABEL = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8"}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ===========================================================================
# Dataset loading (whole images only, no labels folder needed)
# ===========================================================================

def load_dataset(dataset_dir):
    """Folder mode: Fire/images -> 1, NoFire/ -> 0."""
    samples = []
    fire_images_dir = os.path.join(dataset_dir, "Fire", "images")
    nofire_images_dir = os.path.join(dataset_dir, "NoFire")

    if os.path.isdir(fire_images_dir):
        for fname in sorted(os.listdir(fire_images_dir)):
            if fname.lower().endswith(IMG_EXTS):
                samples.append({
                    "image_path": os.path.join(fire_images_dir, fname),
                    "image_name": fname,
                    "label": 1,
                })
    if os.path.isdir(nofire_images_dir):
        for fname in sorted(os.listdir(nofire_images_dir)):
            if fname.lower().endswith(IMG_EXTS):
                samples.append({
                    "image_path": os.path.join(nofire_images_dir, fname),
                    "image_name": fname,
                    "label": 0,
                })
    return samples


def load_dataset_from_csv(images_dir, labels_csv, test_datasets=None):
    """labels.csv mode (same as training / test.py).
    labels.csv columns: id, dataset, fire.  fire==1 -> label 1, else 0.
    Optionally keep only rows whose 'dataset' is in test_datasets."""
    df = pd.read_csv(labels_csv)
    df["fire"] = df["fire"].fillna(0).astype(int)

    if test_datasets:
        df = df[df["dataset"].isin(test_datasets)]

    samples = []
    for _, row in df.iterrows():
        image_path = os.path.join(images_dir, str(row["id"]))
        if not os.path.exists(image_path):
            continue
        samples.append({
            "image_path": image_path,
            "image_name": str(row["id"]),
            "label": 1 if int(row["fire"]) == 1 else 0,
        })
    return samples


def preprocess(image_path, net_size):
    """Whole image -> RGB -> resize net_size -> CHW float32 -> /255 -> ImageNet norm.
    Returns a (1, 3, net_size, net_size) float32 array, or None if unreadable."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (net_size, net_size), interpolation=cv2.INTER_LINEAR)
    chw = resized.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]  # (1,3,H,W)
    chw /= 255.0
    chw = (chw - MEAN) / STD
    return np.ascontiguousarray(chw, dtype=np.float32)


# ===========================================================================
# Backends
# ===========================================================================

class TorchClassifier:
    """FP32 PyTorch classifier — rebuilds the 2-class head like ScanManager."""

    def __init__(self, model_name, weights_path):
        import torch
        from torchvision import models
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = getattr(models, model_name)(weights=None)
        num_classes = 2
        if "resnet" in model_name.lower():
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif "vgg" in model_name.lower() or "alexnet" in model_name.lower():
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        elif "mobilenet" in model_name.lower():
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()
        self.model = model

    def infer(self, np_input):
        """Return logits (1, num_classes). Times only the forward pass via caller."""
        t = self.torch.from_numpy(np_input).to(self.device)
        with self.torch.no_grad():
            out = self.model(t)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        return out.detach().cpu().numpy()


class TRTClassifier:
    """FP16 / INT8 TensorRT classifier — wraps the project's TRTInference."""

    def __init__(self, engine_path):
        from wildfire_detector.TensorRT_infer import TRTInference
        self.engine = TRTInference(engine_path)

    def infer(self, np_input):
        # TRTInference.infer synchronizes its stream internally
        return self.engine.infer(np_input)


def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plain single-image classifier latency/accuracy benchmark (5.3 / 5.5)."
    )
    parser.add_argument("--model_name", required=True,
                        help="torchvision arch: resnet18, resnet34, resnet50, alexnet, vgg16, mobilenet_v2")
    parser.add_argument("--precision", required=True, choices=["fp32", "fp16", "int8"],
                        help="fp32 = PyTorch .pt, fp16/int8 = TensorRT .trt engine")
    parser.add_argument("--weights", required=True,
                        help="Path to .pt (fp32) or .trt engine (fp16/int8).")
    # --- Dataset: choose EITHER folder mode (--dataset) OR labels.csv mode ---
    parser.add_argument("--dataset", default=None,
                        help="Folder mode: dir with Fire/images and NoFire/.")
    parser.add_argument("--images_dir", default=None,
                        help="labels.csv mode: dir with flat images referenced by id.")
    parser.add_argument("--labels_csv", default=None,
                        help="labels.csv mode: CSV with columns id, dataset, fire.")
    parser.add_argument("--test_datasets", nargs="+", default=None,
                        help="labels.csv mode: keep only these source datasets (optional).")
    parser.add_argument("--output_dir", default="benchmark_results",
                        help="Output folder (default: benchmark_results).")
    parser.add_argument("--net_size", type=int, default=224,
                        help="Network input size (default: 224).")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup iterations to discard before timing (default: 20).")
    args = parser.parse_args()

    backend = PRECISION_TO_BACKEND[args.precision]
    precision_label = PRECISION_LABEL[args.precision]

    # --- Build backend ---
    print(f"Loading {args.model_name} ({precision_label}) from {args.weights} ...")
    if args.precision == "fp32":
        clf = TorchClassifier(args.model_name, args.weights)
    else:
        clf = TRTClassifier(args.weights)

    # --- Load dataset (folder mode or labels.csv mode) ---
    if args.labels_csv and args.images_dir:
        samples = load_dataset_from_csv(args.images_dir, args.labels_csv, args.test_datasets)
    elif args.dataset:
        samples = load_dataset(args.dataset)
    else:
        print("ERROR: provide either --dataset (folder mode) or "
              "--images_dir + --labels_csv (labels.csv mode).")
        sys.exit(1)
    fire_count = sum(1 for s in samples if s["label"] == 1)
    print(f"Dataset: {len(samples)} images (Fire: {fire_count}, NoFire: {len(samples) - fire_count})")
    if not samples:
        print("ERROR: no images found. Check --dataset structure (Fire/images + NoFire).")
        sys.exit(1)

    # --- Warmup (discarded) ---
    warm_input = preprocess(samples[0]["image_path"], args.net_size)
    if warm_input is not None:
        for _ in range(args.warmup):
            clf.infer(warm_input)

    # --- Timed inference loop (batch = 1, single forward pass per image) ---
    print("\nRunning single-image inference...")
    results = []
    for sample in tqdm(samples, desc="Inference", unit="img"):
        np_input = preprocess(sample["image_path"], args.net_size)
        if np_input is None:
            continue

        t0 = time.perf_counter()
        logits = clf.infer(np_input)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        probs = softmax(logits)[0]
        pred_label = int(np.argmax(probs))
        results.append({
            "image": sample["image_name"],
            "true_label": sample["label"],
            "pred_label": pred_label,
            "confidence": round(float(probs[pred_label]), 4),
            "model_inference_ms": round(infer_ms, 3),
        })

    df = pd.DataFrame(results)

    # --- Metrics ---
    true_labels = df["true_label"].values
    pred_labels = df["pred_label"].values
    accuracy = (true_labels == pred_labels).mean()
    TP = int(((pred_labels == 1) & (true_labels == 1)).sum())
    FP = int(((pred_labels == 1) & (true_labels == 0)).sum())
    FN = int(((pred_labels == 0) & (true_labels == 1)).sum())
    TN = int(((pred_labels == 0) & (true_labels == 0)).sum())
    fire_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    fire_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fire_f1 = (2 * fire_precision * fire_recall / (fire_precision + fire_recall)
               if (fire_precision + fire_recall) > 0 else 0.0)
    fpr_val = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr_val = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    lat = df["model_inference_ms"]

    # --- Print summary ---
    print("\n" + "=" * 60)
    print(f"CLASSIFIER BENCHMARK — {args.model_name} ({precision_label})")
    print("=" * 60)
    print(f"  Images:                {len(df)}")
    print(f"  Accuracy:              {accuracy*100:.2f}%")
    print(f"  Fire Precision:        {fire_precision*100:.2f}%")
    print(f"  Fire Recall:           {fire_recall*100:.2f}%")
    print(f"  Fire F1:               {fire_f1*100:.2f}%")
    print(f"  FPR / FNR:             {fpr_val*100:.2f}% / {fnr_val*100:.2f}%")
    print(f"  Confusion:  TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    print(f"  ---")
    print(f"  Inference [ms]  mean={lat.mean():.3f}  std={lat.std():.3f}  "
          f"min={lat.min():.3f}  p95={np.percentile(lat, 95):.3f}  max={lat.max():.3f}")
    print(f"  Throughput:            {1000.0 / lat.mean():.1f} img/s")
    print("=" * 60)

    # --- Save (same layout + schema as benchmark_inference.py) ---
    save_dir = os.path.join(args.output_dir, f"{args.model_name}_{backend}")
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    summary = {
        "model": args.model_name,
        "backend": backend,
        "precision": precision_label,
        "num_crops": 1,
        "dataset": args.dataset if args.dataset else args.images_dir,
        "num_images": len(df),
        "accuracy": round(accuracy * 100, 2),
        "fire_precision": round(fire_precision * 100, 2),
        "fire_recall": round(fire_recall * 100, 2),
        "fire_f1": round(fire_f1 * 100, 2),
        "fpr": round(fpr_val * 100, 2),
        "fnr": round(fnr_val * 100, 2),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "mean_model_inference_ms": round(float(lat.mean()), 3),
        "std_model_inference_ms": round(float(lat.std()), 3),
        "min_model_inference_ms": round(float(lat.min()), 3),
        "p95_model_inference_ms": round(float(np.percentile(lat, 95)), 3),
        "max_model_inference_ms": round(float(lat.max()), 3),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "summary.csv"), index=False)
    print(f"\nResults saved to: {save_dir}/")


if __name__ == "__main__":
    main()
