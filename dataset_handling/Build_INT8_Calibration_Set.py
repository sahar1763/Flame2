"""
Build_INT8_Calibration_Set.py
------------------------------
Creates an INT8 calibration image set for TensorRT quantization.

Samples X random Fire images and X random NoFire images from the TRAINING pool
of the UnifiedDataset (i.e. everything EXCEPT the test_only and ignored_datasets
listed in configTrainModel.yaml), then copies them into a flat folder:

    <project_root>/calib_INT8/

Why the training pool only:
  INT8 calibration influences the quantized model, so calibration images must be
  data the model was allowed to see (train/val). Using test_only / ignored
  datasets would leak the test set and invalidate reported accuracy.

Source of truth:
  - configTrainModel.yaml -> dataset.images_dir, dataset.labels_csv,
                             dataset.test_only, dataset.ignored_datasets
  - UnifiedDataset/labels.csv -> columns: id, dataset, fire

Usage:
  python Build_INT8_Calibration_Set.py                 # default 400 per class
  python Build_INT8_Calibration_Set.py --num 500
  python Build_INT8_Calibration_Set.py --num 500 --seed 123 --clean
"""

import os
import shutil
import argparse
import random

import pandas as pd
import yaml


# Resolve important locations relative to this script so it works from anywhere.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "training_framework", "configTrainModel.yaml")
CONFIG_DIR = os.path.dirname(CONFIG_PATH)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "calib_INT8")


def resolve_path(path):
    """Resolve a config path (which may be relative to the config file) to absolute."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(CONFIG_DIR, path))


def main():
    parser = argparse.ArgumentParser(
        description="Build an INT8 calibration set from UnifiedDataset training pool."
    )
    parser.add_argument("--num", type=int, default=400,
                        help="Number of images per class (Fire and NoFire). Default: 400")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling. Default: 42")
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing calib_INT8 folder before writing.")
    args = parser.parse_args()

    # --- Load config ---
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ds_cfg = config.get("dataset", {})
    images_dir = resolve_path(ds_cfg["images_dir"])
    labels_csv = resolve_path(ds_cfg["labels_csv"])
    test_only = ds_cfg.get("test_only", []) or []
    ignored = ds_cfg.get("ignored_datasets", []) or []

    # test_only takes precedence if a dataset appears in both (mirror datasets.py).
    ignored = [d for d in ignored if d not in test_only]
    excluded = set(test_only) | set(ignored)

    print(f"Images dir:       {images_dir}")
    print(f"Labels CSV:       {labels_csv}")
    print(f"Excluded (test):  {sorted(test_only)}")
    print(f"Excluded (ignore):{sorted(ignored)}")

    # --- Load labels and build the training pool ---
    df = pd.read_csv(labels_csv)
    df["fire"] = df["fire"].fillna(0).astype(int)

    # Keep only the training pool: drop test_only + ignored datasets.
    pool = df[~df["dataset"].isin(excluded)].copy()

    # Keep only rows whose image actually exists on disk.
    pool["image_path"] = pool["id"].apply(lambda x: os.path.join(images_dir, str(x)))
    pool = pool[pool["image_path"].apply(os.path.exists)]

    fire_pool = pool[pool["fire"] == 1]
    nofire_pool = pool[pool["fire"] == 0]

    print(f"\nTraining pool available -> Fire: {len(fire_pool)}  NoFire: {len(nofire_pool)}")

    # --- Sample X per class ---
    def sample(frame, n, class_name):
        if len(frame) < n:
            print(f"WARNING: requested {n} {class_name} images but only "
                  f"{len(frame)} available. Using all {len(frame)}.")
            n = len(frame)
        return frame.sample(n=n, random_state=args.seed)

    fire_sel = sample(fire_pool, args.num, "Fire")
    nofire_sel = sample(nofire_pool, args.num, "NoFire")

    # --- Prepare output folder ---
    if args.clean and os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Copy files (prefix with class to avoid any name collision) ---
    copied = 0
    for _, row in pd.concat([fire_sel, nofire_sel]).iterrows():
        label = "fire" if row["fire"] == 1 else "nofire"
        dst_name = f"{label}_{row['dataset']}_{row['id']}"
        dst_path = os.path.join(OUTPUT_DIR, dst_name)
        shutil.copy2(row["image_path"], dst_path)
        copied += 1

    print("\n" + "=" * 55)
    print("INT8 CALIBRATION SET READY")
    print("=" * 55)
    print(f"  Fire images:   {len(fire_sel)}")
    print(f"  NoFire images: {len(nofire_sel)}")
    print(f"  Total copied:  {copied}")
    print(f"  Output folder: {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
