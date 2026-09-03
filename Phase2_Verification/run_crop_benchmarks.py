"""
run_crop_benchmarks.py
----------------------
Automation wrapper that runs the FULL crop-pipeline benchmark
(benchmark_inference.py) across MANY trained models, MANY datasets and the
three crop strategies (raw / single-crop / three-crop) on the Jetson Orin.

This is Report Section 5.6/5.7 -- the "does cropping around the bbox help?"
comparison. Unlike the plain classifier benchmark (run_classifier_benchmarks.py),
benchmark_inference.py loads its weights from the wildfire_detector/ PACKAGE
folder (best_model.pt / best_model_fp16.trt / best_model_int8.trt) and reads the
architecture from wildfire_detector/config.yaml -> phase2.model_name. So for each
model this wrapper first STAGES that model's weights into the package (and patches
config.yaml model_name), runs the benchmarks, then RESTORES the original package
files afterwards so your deployed model is left untouched.

For every model x backend x dataset it runs three crop strategies:
    raw     : whole frame resized to net input, no cropping   (--crop_mode raw)
    crops1  : single sqrt(2) crop around the bbox             (--crop_factors 1.4142)
    crops3  : three-scale crop + majority vote                (--crop_factors 1.2247,1.4142,2.0)

Only CSV results are produced (predictions.csv + summary.csv per run). Figures
and the per-image success/failure visualizations are generated LATER on the PC
from these CSVs (generate_figures.py / visualize_predictions.py).

Output layout (generate_figures.py + visualize_predictions.py both consume it):
    <output_dir>/<dataset_name>/<model>_<backend>_raw/{predictions,summary}.csv
    <output_dir>/<dataset_name>/<model>_<backend>_crops1/{predictions,summary}.csv
    <output_dir>/<dataset_name>/<model>_<backend>_crops3/{predictions,summary}.csv
    <output_dir>/combined_crop_summary.csv     (all runs aggregated)

Dataset root: a folder containing per-source subfolders, each with the standard
Fire/NoFire structure (same as Seperated_Dataset), e.g.
    <dataset_root>/FireSmokeDataset/{Fire/images, Fire/labels_biggest, NoFire}
    <dataset_root>/FireSmokeNEWdataset/{...}
    <dataset_root>/IAI_Datasets/{...}
Fire bboxes are read from --fire_labels_subdir (default 'labels_biggest', because
that is what the sampled/separated datasets ship -- the plain 'labels' folder does
not exist and every fire image would be skipped).

Usage (run FROM Phase2_Verification/ on the Orin):

  # ---- FULL MATRIX (the command actually used for the report) --------------
  # 6 models x 3 backends x 3 datasets x 3 strategies = 162 runs.
  python run_crop_benchmarks.py \
      --dataset_root ../Sampled_Seperated_Dataset \
      --trained_models ../Trained_Models/experiments_2nd \
      --backends pytorch tensorrt int8 \
      --strategies raw crops1 crops3 \
      --fire_labels_subdir labels_biggest \
      --calib_dir ../Sampled_Seperated_Dataset \
      --num_calib 1000 \
      --output_dir crop_results \
      --trtexec /usr/src/tensorrt/bin/trtexec

  # ---- SMALLEST SMOKE TEST (default backends/strategies) -------------------
  # all 6 models, PyTorch FP32 only, all three crop strategies, all datasets.
  python run_crop_benchmarks.py \
      --dataset_root ../Sampled_Seperated_Dataset \
      --output_dir crop_results

  # ---- COMMON SUBSETS ------------------------------------------------------
  # restrict to specific sub-datasets and add the INT8 backend:
  python run_crop_benchmarks.py \
      --dataset_root ../Sampled_Seperated_Dataset \
      --datasets FireSmokeDataset FireSmokeNEWdataset IAI_Datasets \
      --backends pytorch int8 --calib_dir ../Sampled_Seperated_Dataset \
      --num_calib 1000 --output_dir crop_results

  # single model, single strategy (quick check):
  python run_crop_benchmarks.py \
      --dataset_root ../Sampled_Seperated_Dataset \
      --models resnet18 --backends tensorrt --strategies crops3 \
      --output_dir crop_results

  # preview the full plan without executing anything:
  python run_crop_benchmarks.py --dataset_root ../Sampled_Seperated_Dataset --dry_run

Argument combinations (all optional except --dataset_root):
  --models        subset of {alexnet vgg16 resnet18 resnet34 resnet50 mobilenet_v2}
                  (default: all 6)
  --backends      subset of {pytorch(FP32) tensorrt(FP16) int8(INT8)}  (default: pytorch)
  --strategies    subset of {raw crops1 crops3}                        (default: all 3)
                    raw    = whole frame, no crop
                    crops1 = single sqrt(2) crop around the bbox
                    crops3 = three-scale crop + majority vote
  --datasets      subset of the sub-folders under --dataset_root       (default: all)
  --calib_dir     REQUIRED when 'int8' is in --backends. Point at a FOLDER that
                  contains BOTH Fire and NoFire images (recursively globbed) so the
                  INT8 calibration sees both classes; a Fire-only folder biases the
                  quantization scales toward fire.
  --num_calib     number of INT8 calibration images (default: 500; report used 1000)
  --net_size      network input size (default: 224)
  --trtexec       path to trtexec (default: /usr/src/tensorrt/bin/trtexec)
  --no_restore    leave the last-staged model in the package (skip restore)
  --dry_run       print every command without running it

Then, back on the PC, from these CSVs:
  python generate_figures.py --results_dir crop_results --output_dir figures_crop \
      --weights_dir ../Trained_Models/experiments_2nd
  python visualize_predictions.py --dataset ../Sampled_Seperated_Dataset/FireSmokeDataset \
      --predictions crop_results/FireSmokeDataset/mobilenet_v2_pytorch_crops1/predictions.csv --errors
"""

import os
import sys
import time
import glob
import shutil
import argparse
import subprocess

import pandas as pd

DEFAULT_MODELS = ["alexnet", "vgg16", "resnet18", "resnet34", "resnet50", "mobilenet_v2"]
DEFAULT_BACKENDS = ["pytorch"]  # add "tensorrt" (FP16) and/or "int8" as needed

# crop strategy -> (crop_mode, crop_factors string or None, output suffix used by
# benchmark_inference.py so we can find the resulting folder)
CROP_STRATEGIES = {
    "raw":    ("raw",   None,                    "_raw"),
    "crops1": ("crops", "1.4142",                "_crops1"),
    "crops3": ("crops", "1.2247,1.4142,2.0",     "_crops3"),
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))     # .../Phase2_Verification
REPO_ROOT = os.path.dirname(SCRIPT_DIR)                     # repo root
PKG_DIR = os.path.join(REPO_ROOT, "wildfire_detector")
CONFIG_PATH = os.path.join(PKG_DIR, "config.yaml")

# Files inside the package that staging touches (and that we back up / restore).
PKG_WEIGHT_FILES = [
    "best_model.pt",
    "best_model.onnx",
    "best_model_fp16.trt",
    "best_model_int8.trt",
]


# ---------------------------------------------------------------------------
# Model folder discovery
# ---------------------------------------------------------------------------

def find_model_dir(trained_models_dir, model_name):
    """Return the experiment folder whose name ends with '_<model_name>'.

    If several match (multiple timestamps), the newest (largest name) is used.
    """
    suffix = f"_{model_name}"
    matches = [
        d for d in glob.glob(os.path.join(trained_models_dir, "*"))
        if os.path.isdir(d) and os.path.basename(d).endswith(suffix)
    ]
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.basename(p))
    return matches[-1]


def discover_datasets(dataset_root, explicit):
    """Return list of (name, path) sub-datasets with a Fire/ or NoFire/ folder."""
    if explicit:
        out = []
        for name in explicit:
            path = os.path.join(dataset_root, name)
            if os.path.isdir(path):
                out.append((name, path))
            else:
                print(f"  WARN: dataset '{name}' not found under {dataset_root}")
        return out
    out = []
    for entry in sorted(glob.glob(os.path.join(dataset_root, "*"))):
        if os.path.isdir(entry) and (
            os.path.isdir(os.path.join(entry, "Fire"))
            or os.path.isdir(os.path.join(entry, "NoFire"))
        ):
            out.append((os.path.basename(entry), entry))
    return out


# ---------------------------------------------------------------------------
# Package staging (backup / stage / restore)
# ---------------------------------------------------------------------------

def backup_package(backup_dir):
    """Copy the current package weights + config.yaml into backup_dir."""
    os.makedirs(backup_dir, exist_ok=True)
    for fname in PKG_WEIGHT_FILES + ["config.yaml"]:
        src = os.path.join(PKG_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup_dir, fname))
    # Marker files that existed, so restore can delete ones we created.
    with open(os.path.join(backup_dir, "_existed.txt"), "w") as f:
        for fname in PKG_WEIGHT_FILES + ["config.yaml"]:
            if os.path.exists(os.path.join(PKG_DIR, fname)):
                f.write(fname + "\n")


def restore_package(backup_dir):
    """Restore package files from backup_dir; remove files we added."""
    existed = set()
    marker = os.path.join(backup_dir, "_existed.txt")
    if os.path.exists(marker):
        with open(marker) as f:
            existed = {line.strip() for line in f if line.strip()}
    for fname in PKG_WEIGHT_FILES + ["config.yaml"]:
        dst = os.path.join(PKG_DIR, fname)
        backup = os.path.join(backup_dir, fname)
        if fname in existed:
            if os.path.exists(backup):
                shutil.copy2(backup, dst)
        else:
            # File did not exist originally -> remove anything staging created.
            if os.path.exists(dst):
                os.remove(dst)


def patch_config_model_name(model_name):
    """Rewrite the 'model_name:' line under phase2 in config.yaml (keeps comments)."""
    with open(CONFIG_PATH, "r") as f:
        lines = f.readlines()
    changed = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("model_name:"):
            indent = line[: len(line) - len(stripped)]
            comment = ""
            if "#" in line:
                comment = "  #" + line.split("#", 1)[1].rstrip("\n")
            lines[i] = f'{indent}model_name: "{model_name}"{comment}\n'
            changed = True
            break
    if not changed:
        raise RuntimeError("Could not find 'model_name:' line in config.yaml")
    with open(CONFIG_PATH, "w") as f:
        f.writelines(lines)


def stage_model(model_dir, model_name, backends, calib_dir, num_calib, net_size,
                trtexec, dry_run):
    """Stage one model's weights into the package and (re)build engines as needed.

    Returns True on success. Removes stale .trt engines so they are rebuilt for
    the NEW model instead of silently reusing a previous model's engine.
    """
    ckpt = os.path.join(model_dir, "checkpoints")
    pt_src = os.path.join(ckpt, "best_model.pt")
    onnx_src = os.path.join(ckpt, "best_model.onnx")

    print(f"  [stage] {model_name} <- {ckpt}")
    if dry_run:
        print(f"    copy {pt_src} -> {os.path.join(PKG_DIR, 'best_model.pt')}")
        print(f"    copy {onnx_src} -> {os.path.join(PKG_DIR, 'best_model.onnx')}")
        print(f"    patch config.yaml model_name -> {model_name}")
        print("    remove stale best_model_fp16.trt / best_model_int8.trt")
        if "int8" in backends:
            print(f"    build INT8 engine from onnx (calib_dir={calib_dir})")
        return True

    if not os.path.exists(pt_src):
        print(f"    ERROR: missing {pt_src}")
        return False
    shutil.copy2(pt_src, os.path.join(PKG_DIR, "best_model.pt"))
    if os.path.exists(onnx_src):
        shutil.copy2(onnx_src, os.path.join(PKG_DIR, "best_model.onnx"))
    elif set(backends) & {"tensorrt", "int8"}:
        print(f"    ERROR: missing {onnx_src} (needed for TRT engines). "
              f"Run TensorRT_Conversion.py --experiment {model_dir} first.")
        return False

    patch_config_model_name(model_name)

    # Remove stale engines so they rebuild for THIS model.
    for eng in ("best_model_fp16.trt", "best_model_int8.trt"):
        p = os.path.join(PKG_DIR, eng)
        if os.path.exists(p):
            os.remove(p)

    # FP16 is auto-built by ScanManager on first launch. INT8 must be pre-built.
    if "int8" in backends:
        # Per-model calibration cache (kept in the model's checkpoints/ folder).
        # The package engine path is the same for every model, so the cache must
        # be keyed on the model to avoid reusing another model's INT8 scales.
        cache_file = os.path.join(ckpt, "best_model_int8.calib_cache")
        if not build_int8_engine(os.path.join(PKG_DIR, "best_model.onnx"),
                                 os.path.join(PKG_DIR, "best_model_int8.trt"),
                                 calib_dir, net_size, num_calib, cache_file):
            return False
    return True


def build_int8_engine(onnx_path, engine_path, calib_dir, net_size, num_calib, cache_file):
    """Build an INT8 engine, using a fresh per-model calibration cache.

    build_int8_engine.py reuses an existing calibration cache if it finds one, so
    the cache MUST be unique per model and cleared before building. Otherwise the
    first model's INT8 scales would be silently applied to every other model.
    """
    if not calib_dir:
        print("    [int8] ERROR: --calib_dir is required to build the INT8 engine.")
        return False
    # Resolve to absolute paths: build_int8_engine.py runs with cwd=REPO_ROOT.
    onnx_path = os.path.abspath(onnx_path)
    engine_path = os.path.abspath(engine_path)
    calib_dir = os.path.abspath(calib_dir)
    if not os.path.isfile(onnx_path):
        print(f"    [int8] ERROR: ONNX file not found: {onnx_path}")
        return False
    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "build_int8_engine.py"),
        "--onnx", onnx_path,
        "--calib_dir", calib_dir,
        "--output", engine_path,
        "--net_size", str(net_size),
        "--num_calib", str(num_calib),
        "--cache_file", cache_file,
    ]
    print("    [int8] " + " ".join(cmd))
    # Delete any stale calibration cache so this model is calibrated fresh.
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"    [int8] removed stale calibration cache: {cache_file}")
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print("    [int8] BUILD FAILED.")
        return False
    print(f"    [int8] built in {time.perf_counter() - start:.1f}s")
    return True


# ---------------------------------------------------------------------------
# Benchmark invocation
# ---------------------------------------------------------------------------

def run_benchmark(backend, dataset_path, dataset_out_dir, strategy, fire_subdir,
                  dry_run):
    """Invoke benchmark_inference.py for one backend x dataset x crop strategy."""
    crop_mode, crop_factors, _suffix = CROP_STRATEGIES[strategy]
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "benchmark_inference.py"),
        "--dataset", dataset_path,
        "--backend", backend,
        "--output_dir", dataset_out_dir,
        "--crop_mode", crop_mode,
        "--fire_labels_subdir", fire_subdir,
    ]
    if crop_factors is not None:
        cmd += ["--crop_factors", crop_factors]

    print("    [run ] " + " ".join(cmd))
    if dry_run:
        return True
    # benchmark_inference.py does sys.path.append(abspath("..")), so it must run
    # with CWD = Phase2_Verification for `import wildfire_detector`.
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"    [run ] FAILED: {backend} / {strategy} / {dataset_path}")
        return False
    return True


def aggregate(output_dir):
    """Concatenate every summary.csv under output_dir into one CSV."""
    rows = []
    for root, _dirs, files in os.walk(output_dir):
        if "summary.csv" in files:
            df = pd.read_csv(os.path.join(root, "summary.csv"))
            df["dataset_folder"] = os.path.basename(os.path.dirname(root))
            df["run_folder"] = os.path.basename(root)
            rows.append(df)
    if not rows:
        print("No summary.csv files found to aggregate.")
        return
    combined = pd.concat(rows, ignore_index=True)
    out = os.path.join(output_dir, "combined_crop_summary.csv")
    combined.to_csv(out, index=False)
    print(f"\nCombined summary ({len(combined)} rows) -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark_inference.py (crop pipeline) across many "
                    "models x datasets x crop strategies."
    )
    parser.add_argument("--dataset_root", required=True,
                        help="Folder containing per-source sub-datasets "
                             "(each with Fire/NoFire).")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Restrict to these sub-dataset names (default: all "
                             "sub-folders that contain Fire/ or NoFire/).")
    parser.add_argument("--trained_models", default=os.path.join(REPO_ROOT, "Trained_Models"),
                        help="Folder with <timestamp>_<model> experiment folders.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Model names (default: {DEFAULT_MODELS}).")
    parser.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS,
                        choices=["pytorch", "tensorrt", "int8"],
                        help="Backends (default: pytorch). pytorch=FP32, "
                             "tensorrt=FP16, int8=INT8.")
    parser.add_argument("--strategies", nargs="+", default=list(CROP_STRATEGIES.keys()),
                        choices=list(CROP_STRATEGIES.keys()),
                        help="Crop strategies (default: raw crops1 crops3).")
    parser.add_argument("--fire_labels_subdir", default="labels_biggest",
                        help="Fire bbox label subfolder (default: labels_biggest).")
    parser.add_argument("--calib_dir", default=None,
                        help="Calibration image folder for INT8 engine building.")
    parser.add_argument("--num_calib", type=int, default=500,
                        help="Number of calibration images for INT8 (default: 500).")
    parser.add_argument("--net_size", type=int, default=224,
                        help="Network input size (default: 224).")
    parser.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec",
                        help="Path to trtexec (Jetson default).")
    parser.add_argument("--output_dir", default="crop_results",
                        help="Output folder for results (default: crop_results).")
    parser.add_argument("--no_restore", action="store_true",
                        help="Do NOT restore the original package weights/config "
                             "after running (leaves the last-staged model in place).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the commands without executing them.")
    args = parser.parse_args()

    # Absolute paths (subprocesses run with a different CWD).
    args.dataset_root = os.path.abspath(args.dataset_root)
    args.trained_models = os.path.abspath(args.trained_models)
    args.output_dir = os.path.abspath(args.output_dir)
    if args.calib_dir:
        args.calib_dir = os.path.abspath(args.calib_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    datasets = discover_datasets(args.dataset_root, args.datasets)
    if not datasets and not args.dry_run:
        parser.error(f"No sub-datasets with Fire/NoFire found under {args.dataset_root}")

    total = len(args.models) * len(args.backends) * len(datasets) * len(args.strategies)
    print("=" * 70)
    print(f"Crop benchmark matrix: {len(args.models)} models x "
          f"{len(args.backends)} backends x {len(datasets)} datasets x "
          f"{len(args.strategies)} strategies = {total} runs")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Datasets:     {[n for n, _ in datasets]}")
    print(f"Output:       {args.output_dir}")
    print(f"Fire labels:  {args.fire_labels_subdir}")
    print("=" * 70)

    # Back up the current package so we can restore it afterwards.
    backup_dir = os.path.join(args.output_dir, "_package_backup")
    if not args.dry_run:
        backup_package(backup_dir)
        print(f"Backed up package weights/config -> {backup_dir}")

    failures = []
    run_idx = 0
    try:
        for model in args.models:
            print(f"\n### {model}")
            model_dir = find_model_dir(args.trained_models, model)
            if model_dir is None:
                print(f"  SKIP: no folder ending with '_{model}' in {args.trained_models}")
                run_idx += len(args.backends) * len(datasets) * len(args.strategies)
                failures.append(f"{model} (no experiment folder)")
                continue
            print(f"  folder: {model_dir}")

            if not stage_model(model_dir, model, args.backends, args.calib_dir,
                               args.num_calib, args.net_size, args.trtexec,
                               args.dry_run):
                run_idx += len(args.backends) * len(datasets) * len(args.strategies)
                failures.append(f"{model} (staging failed)")
                continue

            for backend in args.backends:
                for ds_name, ds_path in (datasets or [("<dataset>", args.dataset_root)]):
                    ds_out = os.path.join(args.output_dir, ds_name)
                    for strategy in args.strategies:
                        run_idx += 1
                        print(f"\n  --- [{run_idx}/{total}] {model} / {backend} / "
                              f"{ds_name} / {strategy} ---")
                        ok = run_benchmark(backend, ds_path, ds_out, strategy,
                                           args.fire_labels_subdir, args.dry_run)
                        if not ok:
                            failures.append(f"{model}/{backend}/{ds_name}/{strategy}")
    finally:
        if not args.dry_run:
            if args.no_restore:
                print("\n--no_restore set: leaving last-staged model in the package.")
            else:
                restore_package(backup_dir)
                print(f"\nRestored original package weights/config from {backup_dir}")

    if not args.dry_run:
        aggregate(args.output_dir)

    print("\n" + "=" * 70)
    if failures:
        print(f"Completed with {len(failures)} problem run(s):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("All runs completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
