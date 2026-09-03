r"""
train_all_models.py
-------------------
Sequential multi-model train + test orchestrator for the wildfire classifier.

Purpose
=======
Trains a list of candidate architectures ONE AT A TIME (matching the
space-constrained Jetson Orin workflow) using the existing DDP training entry
point (train.py), then immediately evaluates each trained model on the held-out
test split using test.py. Finally, it aggregates the per-model 'all' test
summaries into a single CSV that mirrors Table 9 in the thesis report
(Classification performance of candidate models on the held-out test set).

It does NOT reimplement training or evaluation logic. It only:
  1. Writes a per-model config (copy of the base config with model_name overridden).
  2. Launches:  torchrun --standalone --nproc_per_node=N train.py --config <tmp>
  3. Detects the freshly created ../experiments/<timestamp>_<model_name> folder.
  4. Launches:  python test.py --experiment <that folder>
  5. Collects test/all/evaluation_summary.csv from every model into one table.

Because each model is trained in its own subprocess and only best_model.pt is
kept per experiment (train.py already saves just the best checkpoint), disk
usage stays bounded to one model at a time plus the saved checkpoints.

Usage
=====
  # Train + test the full candidate set (report Table 8) on a single GPU:
  python train_all_models.py --config configTrainModel.yaml --nproc 1

  # Only a subset of models:
  python train_all_models.py --models resnet18 mobilenet_v2

  # Train only, skip evaluation:
  python train_all_models.py --no_test

  # Skip models whose experiment folder already exists (resume a broken run):
  python train_all_models.py --skip_existing

  # Preview the commands without running anything:
  python train_all_models.py --dry_run

Notes
=====
  * Run this from the training_framework/ directory (same working directory
    train.py expects, so that ../experiments and ../<dataset> resolve correctly).
  * The dataset path, hyperparameters, test_only and ignored_datasets are all
    taken from the base config unchanged; only training.model_name is overridden
    per model.
  * On the Orin (single GPU) use --nproc 1. train.py always initializes a
    process group, so it must be launched through torchrun even for one GPU.
"""

import os
import sys
import glob
import time
import copy
import argparse
import subprocess
from datetime import datetime

import yaml
import pandas as pd


# Candidate architectures evaluated in the thesis (report Table 8).
DEFAULT_MODELS = [
    "alexnet",
    "vgg16",
    "resnet18",
    "resnet34",
    "resnet50",
    "mobilenet_v2",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_base_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_to_project(model_name):
    """Derive a W&B project name from the model name (e.g. resnet18 -> Phase2_ResNet18)."""
    special = {
        "resnet18": "ResNet18",
        "resnet34": "ResNet34",
        "resnet50": "ResNet50",
        "vgg16": "VGG16",
        "alexnet": "AlexNet",
        "mobilenet_v2": "MobileNetV2",
    }
    pretty = special.get(model_name, model_name)
    return f"Phase2_{pretty}"


def write_model_config(base_config, model_name, tmp_dir):
    """Write a per-model config (base config with model_name + wandb project overridden)."""
    os.makedirs(tmp_dir, exist_ok=True)
    cfg = copy.deepcopy(base_config)
    cfg["training"]["model_name"] = model_name
    # Derive the W&B project from the model name so each architecture logs separately.
    cfg.setdefault("wandb", {})["project"] = model_to_project(model_name)
    out_path = os.path.join(tmp_dir, f"config_{model_name}.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def list_experiment_dirs(experiments_dir):
    if not os.path.isdir(experiments_dir):
        return set()
    return {
        d for d in os.listdir(experiments_dir)
        if os.path.isdir(os.path.join(experiments_dir, d))
    }


def find_new_experiment(experiments_dir, before, model_name):
    """Return the experiment folder created by the training run just finished."""
    after = list_experiment_dirs(experiments_dir)
    created = sorted(after - before)
    # Prefer a folder that ends with the model name (train.py naming convention).
    matching = [d for d in created if d.endswith(f"_{model_name}")]
    candidates = matching or created
    if not candidates:
        return None
    # Newest by modification time.
    candidates.sort(
        key=lambda d: os.path.getmtime(os.path.join(experiments_dir, d))
    )
    return os.path.join(experiments_dir, candidates[-1])


def run_subprocess(cmd, cwd, dry_run):
    print("\n$ " + " ".join(cmd) + f"    (cwd={cwd})", flush=True)
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def train_one(model_name, base_config, args, tmp_dir):
    """Train a single model via torchrun. Returns the experiment dir or None."""
    experiments_dir = os.path.join(SCRIPT_DIR, args.experiments_dir)

    if args.skip_existing:
        existing = [
            d for d in list_experiment_dirs(experiments_dir)
            if d.endswith(f"_{model_name}")
        ]
        if existing:
            existing.sort(
                key=lambda d: os.path.getmtime(os.path.join(experiments_dir, d))
            )
            found = os.path.join(experiments_dir, existing[-1])
            print(f"[skip_existing] {model_name}: reusing {found}")
            return found

    cfg_path = write_model_config(base_config, model_name, tmp_dir)
    before = list_experiment_dirs(experiments_dir)

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc}",
        "train.py",
        "--config", cfg_path,
    ]
    rc = run_subprocess(cmd, cwd=SCRIPT_DIR, dry_run=args.dry_run)
    if rc != 0:
        print(f"[ERROR] Training failed for {model_name} (exit code {rc}).")
        return None

    if args.dry_run:
        return None

    exp_dir = find_new_experiment(experiments_dir, before, model_name)
    if exp_dir is None:
        print(f"[ERROR] Could not locate new experiment folder for {model_name}.")
    else:
        print(f"[ok] {model_name} -> {exp_dir}")
    return exp_dir


def test_one(exp_dir, args):
    """Evaluate a trained model on the held-out test split via test.py."""
    if exp_dir is None:
        return
    cmd = [sys.executable, "test.py", "--experiment", exp_dir]
    rc = run_subprocess(cmd, cwd=SCRIPT_DIR, dry_run=args.dry_run)
    if rc != 0:
        print(f"[ERROR] Testing failed for {exp_dir} (exit code {rc}).")


def collect_summary(model_name, exp_dir, summary_rows):
    """Read test/all/evaluation_summary.csv for one model into the combined table."""
    if exp_dir is None:
        return
    summary_csv = os.path.join(exp_dir, "test", "all", "evaluation_summary.csv")
    if not os.path.isfile(summary_csv):
        print(f"[warn] No test summary for {model_name} at {summary_csv}")
        return
    row = pd.read_csv(summary_csv).iloc[0].to_dict()
    row = {"model": model_name, "experiment": os.path.basename(exp_dir), **row}
    summary_rows.append(row)


def main():
    parser = argparse.ArgumentParser(
        description="Sequentially train + test multiple candidate models."
    )
    parser.add_argument("--config", default="configTrainModel.yaml",
                        help="Base training config (model_name is overridden per model).")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Models to train. Default: {DEFAULT_MODELS}")
    parser.add_argument("--nproc", type=int, default=1,
                        help="GPUs per node for torchrun (Orin single-GPU = 1).")
    parser.add_argument("--experiments_dir", default="../experiments",
                        help="Experiments root, relative to training_framework/.")
    parser.add_argument("--tmp_config_dir", default="_multimodel_configs",
                        help="Directory for the generated per-model configs.")
    parser.add_argument("--no_test", action="store_true",
                        help="Train only; do not run test.py after each model.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Reuse an existing experiment folder for a model if present.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the commands without executing them.")
    args = parser.parse_args()

    base_config = load_base_config(os.path.join(SCRIPT_DIR, args.config)
                                   if not os.path.isabs(args.config) else args.config)
    tmp_dir = os.path.join(SCRIPT_DIR, args.tmp_config_dir)
    experiments_dir = os.path.join(SCRIPT_DIR, args.experiments_dir)

    print("=" * 70)
    print("MULTI-MODEL TRAIN + TEST ORCHESTRATOR")
    print("=" * 70)
    print(f"  Base config : {args.config}")
    print(f"  Models      : {args.models}")
    print(f"  GPUs/node   : {args.nproc}")
    print(f"  Experiments : {experiments_dir}")
    print(f"  Test after  : {not args.no_test}")
    print("=" * 70)

    summary_rows = []
    for i, model_name in enumerate(args.models, 1):
        print(f"\n########## [{i}/{len(args.models)}] {model_name} "
              f"({datetime.now().strftime('%H:%M:%S')}) ##########")
        t0 = time.time()

        exp_dir = train_one(model_name, base_config, args, tmp_dir)
        if not args.no_test:
            test_one(exp_dir, args)
            collect_summary(model_name, exp_dir, summary_rows)

        print(f"[time] {model_name}: {(time.time() - t0) / 60.0:.1f} min")

    # Aggregate the per-model 'all' summaries (report Table 9).
    if summary_rows:
        out_csv = os.path.join(experiments_dir, "all_models_test_summary.csv")
        os.makedirs(experiments_dir, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
        print("\n" + "=" * 70)
        print("COMBINED TEST SUMMARY (held-out test set)")
        print("=" * 70)
        print(pd.DataFrame(summary_rows).to_string(index=False))
        print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
