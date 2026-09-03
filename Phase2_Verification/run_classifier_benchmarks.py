"""
run_classifier_benchmarks.py
----------------------------
Automation wrapper that runs the plain single-image classifier benchmark
(classifier_latency_benchmark.py) across MANY trained models and MANY
deployment precisions on the Jetson Orin, building the TensorRT engines it
needs along the way.

Report activity A ("run the classifier inference on 18 models"):
    6 models  x  3 precisions (fp32 / fp16 / int8)  =  18 benchmark runs.

For every model it:
    1. Locates the model's experiment folder inside --trained_models
       (matched by the folder-name suffix "_<model_name>", e.g.
        2026-08-08_17-08-21_mobilenet_v2 -> mobilenet_v2).
    2. Uses  checkpoints/best_model.pt   for the fp32 (PyTorch) run.
    3. Builds  checkpoints/best_model_fp16.trt  from best_model.onnx via
       trtexec --fp16  (same command ScanManager uses) for the fp16 run.
    4. Builds  checkpoints/best_model_int8.trt  from best_model.onnx via
       build_int8_engine.py (needs --calib_dir) for the int8 run.
    5. Runs classifier_latency_benchmark.py once per precision, pointing
       --weights directly at the .pt / .trt file (no package staging needed).
    6. Aggregates every  <model>_<backend>/summary.csv  into one
       combined_classifier_summary.csv (18 rows) for the report tables.

The same dataset target is applied to all runs (so the 18 numbers are
comparable). Choose EITHER labels.csv mode (recommended, matches training /
test.py) OR folder mode:

  # FULL command as run on the Orin (6 models x 3 precisions = 18 runs).
  # --trained_models points at the folder holding the <timestamp>_<model>
  # experiment folders (here the 2nd training batch, experiments_2nd).
  python run_classifier_benchmarks.py \
      --images_dir ../Resized_TestSet_Cropped_Biggest \
      --labels_csv ../Resized_TestSet_Cropped_Biggest/labels.csv \
      --test_datasets FireSmokeDataset FireSmokeNEWdataset \
      --calib_dir ../Resized_TestSet_Cropped_Biggest \
      --num_calib 1000 \
      --rebuild_engines \
      --output_dir classifier_results \
      --trained_models ../Trained_Models/experiments_2nd

  # folder mode -- a Fire/images + NoFire directory
  python run_classifier_benchmarks.py \
      --dataset ../SomeTestSet \
      --calib_dir ../SomeTestSet \
      --output_dir classifier_results

Argument combinations (all optional except the dataset target):
  --models        subset of {alexnet vgg16 resnet18 resnet34 resnet50 mobilenet_v2}
                  (default: all 6)
  --precisions    subset of {fp32 fp16 int8}                          (default: all 3)
  Dataset target  EITHER  --images_dir + --labels_csv (+ optional --test_datasets
                  to restrict which CSV sources are used)   [recommended, matches
                  training/test.py]   OR   --dataset <folder with Fire/ + NoFire/>.
  --calib_dir     REQUIRED when 'int8' is in --precisions and the engine must be
                  built. Point at a folder of representative images (recursively
                  globbed); include BOTH classes so scales are not fire-biased.
  --num_calib     number of INT8 calibration images (default: 500; report used 1000)
  --net_size      network input size (default: 224)
  --trtexec       path to trtexec (default: /usr/src/tensorrt/bin/trtexec)
  --rebuild_engines   force-rebuild .trt engines even if cached
  --skip_engine_build assume .trt engines already exist (fail if missing)
  --dry_run       print every command without running it

  # single model, single precision (quick check):
  python run_classifier_benchmarks.py \
      --dataset ../SomeTestSet --models resnet18 --precisions int8 \
      --calib_dir ../SomeTestSet --output_dir classifier_results

Notes
-----
* Run this FROM the Phase2_Verification/ folder on the Orin.
* fp16/int8 require tensorrt + trtexec (+ pycuda for int8 calibration), so this
  only runs on the Jetson, not on the Windows dev box. Use --dry_run anywhere
  to print the exact commands without executing them.
* --calib_dir is only needed when 'int8' is in --precisions and the engine has
  to be built. Point it at a folder that contains representative images
  (recursively globbed by build_int8_engine.py).
* Engines are cached in each model's checkpoints/ folder; re-runs reuse them
  unless --rebuild_engines is given.
"""

import os
import sys
import time
import glob
import argparse
import subprocess

import pandas as pd

DEFAULT_MODELS = ["alexnet", "vgg16", "resnet18", "resnet34", "resnet50", "mobilenet_v2"]
DEFAULT_PRECISIONS = ["fp32", "fp16", "int8"]

# precision -> (engine/weight filename, backend label used by the benchmark output folder)
PRECISION_TO_BACKEND = {"fp32": "pytorch", "fp16": "tensorrt", "int8": "int8"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))     # .../Phase2_Verification
REPO_ROOT = os.path.dirname(SCRIPT_DIR)                     # repo root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_model_dir(trained_models_dir, model_name):
    """Return the experiment folder whose name ends with '_<model_name>'.

    If several match (multiple timestamps), the lexicographically largest is
    chosen, which for the YYYY-MM-DD_HH-MM-SS prefix is the newest one.
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


def build_fp16_engine(onnx_path, engine_path, net_size, trtexec, dry_run):
    """Build an FP16 TensorRT engine with trtexec (same cmd as ScanManager)."""
    onnx_path = os.path.abspath(onnx_path)
    engine_path = os.path.abspath(engine_path)
    if not os.path.isfile(onnx_path) and not dry_run:
        print(f"  [fp16] ERROR: ONNX file not found: {onnx_path}")
        return False
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--minShapes=input:1x3x{net_size}x{net_size}",
        f"--optShapes=input:3x3x{net_size}x{net_size}",
        f"--maxShapes=input:16x3x{net_size}x{net_size}",
        f"--shapes=input:1x3x{net_size}x{net_size}",
    ]
    print("  [fp16] " + " ".join(cmd))
    if dry_run:
        return True
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [fp16] BUILD FAILED:\n{result.stderr[-2000:]}")
        return False
    print(f"  [fp16] built in {time.perf_counter() - start:.1f}s -> {engine_path}")
    return True


def build_int8_engine(onnx_path, engine_path, calib_dir, net_size, num_calib, dry_run):
    """Build an INT8 TensorRT engine via build_int8_engine.py.

    Uses a per-model calibration cache (next to the output engine) and DELETES
    any stale cache first. build_int8_engine.py reuses an existing cache if it
    finds one, so a shared/leftover cache would apply one model's INT8 scales to
    a different model. A unique, freshly-cleared cache per model prevents that.
    """
    if not calib_dir:
        print("  [int8] ERROR: --calib_dir is required to build the INT8 engine.")
        return False

    # Resolve to absolute paths: build_int8_engine.py runs with cwd=REPO_ROOT, so
    # any path derived from a relative --trained_models would otherwise break.
    onnx_path = os.path.abspath(onnx_path)
    engine_path = os.path.abspath(engine_path)
    calib_dir = os.path.abspath(calib_dir)

    if not os.path.isfile(onnx_path) and not dry_run:
        print(f"  [int8] ERROR: ONNX file not found: {onnx_path}")
        return False

    # Per-model cache path (engine_path lives in that model's checkpoints/ folder).
    cache_file = os.path.splitext(engine_path)[0] + ".calib_cache"
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "build_int8_engine.py"),
        "--onnx", onnx_path,
        "--calib_dir", calib_dir,
        "--output", engine_path,
        "--net_size", str(net_size),
        "--num_calib", str(num_calib),
        "--cache_file", cache_file,
    ]
    print("  [int8] " + " ".join(cmd))
    if dry_run:
        return True
    # Delete any stale calibration cache so this model is calibrated fresh.
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"  [int8] removed stale calibration cache: {cache_file}")
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print("  [int8] BUILD FAILED.")
        return False
    print(f"  [int8] built in {time.perf_counter() - start:.1f}s -> {engine_path}")
    return True


def run_classifier_benchmark(model_name, precision, weights_path, args):
    """Invoke classifier_latency_benchmark.py for one model x precision."""
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "classifier_latency_benchmark.py"),
        "--model_name", model_name,
        "--precision", precision,
        "--weights", weights_path,
        "--output_dir", args.output_dir,
        "--net_size", str(args.net_size),
    ]
    if args.labels_csv and args.images_dir:
        cmd += ["--images_dir", args.images_dir, "--labels_csv", args.labels_csv]
        if args.test_datasets:
            cmd += ["--test_datasets", *args.test_datasets]
    elif args.dataset:
        cmd += ["--dataset", args.dataset]

    print("  [run ] " + " ".join(cmd))
    if args.dry_run:
        return True
    # classifier_latency_benchmark.py does sys.path.insert(0, abspath("..")),
    # so it must run with CWD = Phase2_Verification for `import wildfire_detector`.
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"  [run ] BENCHMARK FAILED for {model_name} ({precision}).")
        return False
    return True


def aggregate_summaries(output_dir, models, precisions):
    """Concatenate every <model>_<backend>/summary.csv into one CSV."""
    rows = []
    for model in models:
        for precision in precisions:
            backend = PRECISION_TO_BACKEND[precision]
            summary_path = os.path.join(output_dir, f"{model}_{backend}", "summary.csv")
            if os.path.exists(summary_path):
                rows.append(pd.read_csv(summary_path))
    if not rows:
        print("No summary.csv files found to aggregate.")
        return
    combined = pd.concat(rows, ignore_index=True)
    out = os.path.join(output_dir, "combined_classifier_summary.csv")
    combined.to_csv(out, index=False)
    print(f"\nCombined summary ({len(combined)} rows) -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run classifier_latency_benchmark.py across many models x precisions."
    )
    parser.add_argument("--trained_models", default=os.path.join(REPO_ROOT, "Trained_Models"),
                        help="Folder containing <timestamp>_<model> experiment folders.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Model names to benchmark (default: {DEFAULT_MODELS}).")
    parser.add_argument("--precisions", nargs="+", default=DEFAULT_PRECISIONS,
                        choices=["fp32", "fp16", "int8"],
                        help="Precisions to run (default: fp32 fp16 int8).")
    # --- Dataset target (applied to every run): labels.csv mode OR folder mode ---
    parser.add_argument("--dataset", default=None,
                        help="Folder mode: dir with Fire/images and NoFire/.")
    parser.add_argument("--images_dir", default=None,
                        help="labels.csv mode: dir with flat images referenced by id.")
    parser.add_argument("--labels_csv", default=None,
                        help="labels.csv mode: CSV with columns id, dataset, fire.")
    parser.add_argument("--test_datasets", nargs="+", default=None,
                        help="labels.csv mode: keep only these source datasets.")
    # --- Engine building ---
    parser.add_argument("--calib_dir", default=None,
                        help="Calibration image folder for INT8 engine building.")
    parser.add_argument("--num_calib", type=int, default=500,
                        help="Number of calibration images for INT8 (default: 500).")
    parser.add_argument("--net_size", type=int, default=224,
                        help="Network input size (default: 224).")
    parser.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec",
                        help="Path to the trtexec binary (Jetson default).")
    parser.add_argument("--skip_engine_build", action="store_true",
                        help="Do not build engines; reuse existing .trt files.")
    parser.add_argument("--rebuild_engines", action="store_true",
                        help="Rebuild .trt engines even if they already exist.")
    # --- Output / control ---
    parser.add_argument("--output_dir", default="classifier_results",
                        help="Output folder for results (default: classifier_results).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the commands without executing them.")
    args = parser.parse_args()

    # Resolve output_dir to an absolute path so the subprocess CWD change is safe.
    args.trained_models = os.path.abspath(args.trained_models)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate dataset selection.
    labels_mode = bool(args.labels_csv and args.images_dir)
    if not labels_mode and not args.dataset and not args.dry_run:
        parser.error("Provide either --dataset (folder mode) or "
                     "--images_dir + --labels_csv (labels.csv mode).")

    # Make dataset paths absolute (subprocess runs with a different CWD).
    for attr in ("dataset", "images_dir", "labels_csv", "calib_dir"):
        val = getattr(args, attr)
        if val:
            setattr(args, attr, os.path.abspath(val))

    print("=" * 70)
    print(f"Classifier benchmark matrix: {len(args.models)} models x "
          f"{len(args.precisions)} precisions = "
          f"{len(args.models) * len(args.precisions)} runs")
    print(f"Trained models: {args.trained_models}")
    print(f"Output:         {args.output_dir}")
    print("=" * 70)

    total = len(args.models) * len(args.precisions)
    run_idx = 0
    failures = []

    for model in args.models:
        model_dir = find_model_dir(args.trained_models, model)
        print(f"\n### {model}")
        if model_dir is None:
            print(f"  SKIP: no folder ending with '_{model}' in {args.trained_models}")
            for precision in args.precisions:
                failures.append(f"{model}/{precision} (no experiment folder)")
                run_idx += 1
            continue

        ckpt_dir = os.path.join(model_dir, "checkpoints")
        pt_path = os.path.join(ckpt_dir, "best_model.pt")
        onnx_path = os.path.join(ckpt_dir, "best_model.onnx")
        fp16_path = os.path.join(ckpt_dir, "best_model_fp16.trt")
        int8_path = os.path.join(ckpt_dir, "best_model_int8.trt")
        print(f"  folder: {model_dir}")

        for precision in args.precisions:
            run_idx += 1
            print(f"\n  --- [{run_idx}/{total}] {model} / {precision} ---")

            if precision == "fp32":
                weights = pt_path
                if not os.path.exists(pt_path) and not args.dry_run:
                    print(f"  SKIP: missing {pt_path}")
                    failures.append(f"{model}/fp32 (missing .pt)")
                    continue

            elif precision == "fp16":
                weights = fp16_path
                need_build = args.rebuild_engines or not os.path.exists(fp16_path)
                if need_build and not args.skip_engine_build:
                    if not os.path.exists(onnx_path) and not args.dry_run:
                        print(f"  SKIP: missing {onnx_path} (run TensorRT_Conversion.py first)")
                        failures.append(f"{model}/fp16 (missing .onnx)")
                        continue
                    if not build_fp16_engine(onnx_path, fp16_path, args.net_size,
                                             args.trtexec, args.dry_run):
                        failures.append(f"{model}/fp16 (engine build failed)")
                        continue
                elif not os.path.exists(fp16_path) and not args.dry_run:
                    print(f"  SKIP: missing {fp16_path} and engine building disabled")
                    failures.append(f"{model}/fp16 (missing engine)")
                    continue

            else:  # int8
                weights = int8_path
                need_build = args.rebuild_engines or not os.path.exists(int8_path)
                if need_build and not args.skip_engine_build:
                    if not os.path.exists(onnx_path) and not args.dry_run:
                        print(f"  SKIP: missing {onnx_path} (run TensorRT_Conversion.py first)")
                        failures.append(f"{model}/int8 (missing .onnx)")
                        continue
                    if not build_int8_engine(onnx_path, int8_path, args.calib_dir,
                                             args.net_size, args.num_calib, args.dry_run):
                        failures.append(f"{model}/int8 (engine build failed)")
                        continue
                elif not os.path.exists(int8_path) and not args.dry_run:
                    print(f"  SKIP: missing {int8_path} and engine building disabled")
                    failures.append(f"{model}/int8 (missing engine)")
                    continue

            if not run_classifier_benchmark(model, precision, weights, args):
                failures.append(f"{model}/{precision} (benchmark failed)")

    # Aggregate everything that succeeded.
    if not args.dry_run:
        aggregate_summaries(args.output_dir, args.models, args.precisions)

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
