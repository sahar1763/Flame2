r"""
Convert_FireSmoke_Dataset.py
----------------------------
Converts a Roboflow "FireSmokeDataset"-style YOLO dataset into the project's
binary classification folder structure (same layout as the corrected D_Fire
dataset), with unified single-bbox labels.

Source classes (from data.yaml):  names: ['fire', 'other', 'smoke']
  0 = fire
  1 = other   <- ignored (treated as non-fire)
  2 = smoke

Classification rules (per image):
  - Has any fire box (class 0)                 -> Fire/
  - No fire, but has smoke (class 2)           -> SKIP (ignored entirely)
  - No fire, no smoke (only 'other' or empty)  -> NoFire/

Unified bbox (Fire images only):
  - Merge ALL fire + smoke boxes into one enclosing box.
  - 'other' boxes are NOT included in the merge (treated as non-fire).
  - Always written with class ID 1 (fire) in YOLO format.

Output structure (matches corrected D_Fire layout):
  <OUTPUT_ROOT>/
    Fire/
      images/     <- fire images copied here
      labels/     <- one label per image: single enclosing bbox (class=1)
    NoFire/       <- no-fire images copied DIRECTLY here (no images/ subfolder)

Notes:
  - NoFire has no labels and no subfolder (matches the fixed dataset structure).
  - Update SOURCE_ROOT / OUTPUT_ROOT below and run once per dataset.

Label modes (--label_mode):
  - unified      -> merge fire + smoke boxes into one enclosing bbox (default).
  - biggest_fire -> keep only the single largest FIRE box (smoke filtered out).
                    Output auto-suffixed with '_biggestfire' unless --output given.

Usage:
  # UNIFIED (default output = OUTPUT_ROOT)
  python Convert_FireSmoke_Dataset.py --label_mode unified

  # BIGGEST FIRE (auto output = OUTPUT_ROOT + "_biggestfire")
  python Convert_FireSmoke_Dataset.py --label_mode biggest_fire

  # Override raw source / output explicitly
  python Convert_FireSmoke_Dataset.py --label_mode biggest_fire \
    --source "<raw_roboflow_root>" \
    --output "J:\ControlProjects\FireDrone\Seperated_Dataset\FireSmokeDataset_biggestfire"
"""

import os
import shutil
import argparse

# === Configuration (update per dataset) ===
SOURCE_ROOT = r"J:\ControlProjects\FireDrone\Seperated_Dataset\documents_20260808\FireSmokeDataset\FireSmokeDataset\FireSmokeNEWdataset.v1i.yolov5pytorch"
OUTPUT_ROOT = r"J:\ControlProjects\FireDrone\Seperated_Dataset\FireSmokeNEWdataset"

# Splits to look for inside SOURCE_ROOT (missing ones are skipped with a warning).
SPLITS = ["train", "valid", "test"]

# Class IDs in the source dataset
FIRE_CLASS = 0
OTHER_CLASS = 1
SMOKE_CLASS = 2

# Class ID written to the unified output labels
OUTPUT_CLASS_ID = 1

# Default labeling mode:
#   "unified"      -> merge fire + smoke boxes into one enclosing bbox
#   "biggest_fire" -> keep only the single largest fire bbox
LABEL_MODE = "unified"


def parse_yolo_labels(label_path):
    """
    Parse a YOLO label file.
    Returns list of tuples: (class_id, x_center, y_center, width, height)
    """
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            boxes.append((class_id, x_center, y_center, w, h))
    return boxes


def compute_enclosing_bbox(boxes):
    """
    Given a list of YOLO boxes (class_id, xc, yc, w, h), compute the single
    enclosing bounding box in YOLO format.

    Returns: (x_center, y_center, width, height) normalized.
    """
    x_mins = []
    y_mins = []
    x_maxs = []
    y_maxs = []

    for _, xc, yc, w, h in boxes:
        x_mins.append(xc - w / 2)
        y_mins.append(yc - h / 2)
        x_maxs.append(xc + w / 2)
        y_maxs.append(yc + h / 2)

    x_min = min(x_mins)
    y_min = min(y_mins)
    x_max = max(x_maxs)
    y_max = max(y_maxs)

    # Clip to [0, 1]
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    x_max = min(1.0, x_max)
    y_max = min(1.0, y_max)

    enc_w = x_max - x_min
    enc_h = y_max - y_min
    enc_xc = x_min + enc_w / 2
    enc_yc = y_min + enc_h / 2

    return enc_xc, enc_yc, enc_w, enc_h


def select_biggest_fire_bbox(boxes, fire_class):
    """
    Return the single largest FIRE bounding box (by normalized area) as
    (x_center, y_center, width, height). Non-fire boxes are ignored.
    Returns None if no fire box is present.
    """
    fire_boxes = [b for b in boxes if b[0] == fire_class]
    if not fire_boxes:
        return None
    _, xc, yc, w, h = max(fire_boxes, key=lambda b: b[3] * b[4])
    return xc, yc, w, h


def main():
    parser = argparse.ArgumentParser(
        description="Convert a FireSmoke YOLO dataset to binary classification structure.")
    parser.add_argument("--label_mode", choices=["unified", "biggest_fire"],
                        default=LABEL_MODE,
                        help="unified = merge fire + smoke boxes into one enclosing bbox; "
                             "biggest_fire = keep only the largest fire bbox.")
    parser.add_argument("--source", default=SOURCE_ROOT, help="Raw dataset source root.")
    parser.add_argument("--output", default=None,
                        help="Output root. Default: OUTPUT_ROOT, with '_biggestfire' "
                             "suffix in biggest_fire mode.")
    args = parser.parse_args()

    label_mode = args.label_mode
    source_root = args.source
    if args.output:
        output_root = args.output
    else:
        output_root = OUTPUT_ROOT if label_mode == "unified" else OUTPUT_ROOT + "_biggestfire"

    print(f"Label mode: {label_mode}")

    # Create output directories (Fire/images, Fire/labels, NoFire directly)
    fire_images_dir = os.path.join(output_root, "Fire", "images")
    fire_labels_dir = os.path.join(output_root, "Fire", "labels")
    nofire_dir = os.path.join(output_root, "NoFire")

    os.makedirs(fire_images_dir, exist_ok=True)
    os.makedirs(fire_labels_dir, exist_ok=True)
    os.makedirs(nofire_dir, exist_ok=True)

    counts = {"fire": 0, "nofire": 0, "smoke_only_skipped": 0}

    for split in SPLITS:
        images_dir = os.path.join(source_root, split, "images")
        labels_dir = os.path.join(source_root, split, "labels")

        if not os.path.isdir(images_dir):
            print(f"WARNING: {images_dir} not found, skipping.")
            continue

        image_files = sorted([f for f in os.listdir(images_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        print(f"Processing split: {split} ({len(image_files)} images)")

        for img_name in image_files:
            img_path = os.path.join(images_dir, img_name)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_name)

            # Parse label file
            if os.path.exists(label_path):
                boxes = parse_yolo_labels(label_path)
            else:
                boxes = []

            has_fire = any(cls == FIRE_CLASS for cls, *_ in boxes)
            has_smoke = any(cls == SMOKE_CLASS for cls, *_ in boxes)

            if has_fire:
                # Fire image -> copy image + create bbox label.
                shutil.copy2(img_path, os.path.join(fire_images_dir, img_name))

                if label_mode == "unified":
                    # Merge only fire + smoke boxes (ignore 'other').
                    merge_boxes = [b for b in boxes
                                   if b[0] in (FIRE_CLASS, SMOKE_CLASS)]
                    enc_xc, enc_yc, enc_w, enc_h = compute_enclosing_bbox(merge_boxes)
                else:
                    # Keep only the largest fire bbox.
                    enc_xc, enc_yc, enc_w, enc_h = select_biggest_fire_bbox(boxes, FIRE_CLASS)

                out_label_path = os.path.join(fire_labels_dir, label_name)
                with open(out_label_path, "w") as f:
                    f.write(f"{OUTPUT_CLASS_ID} {enc_xc:.6f} {enc_yc:.6f} {enc_w:.6f} {enc_h:.6f}\n")

                counts["fire"] += 1

            elif has_smoke:
                # Smoke only (no fire) -> skip entirely.
                counts["smoke_only_skipped"] += 1

            else:
                # No fire, no smoke (empty or only 'other') -> NoFire.
                shutil.copy2(img_path, os.path.join(nofire_dir, img_name))
                counts["nofire"] += 1

    # --- Summary ---
    print("\n" + "=" * 50)
    print("FIRE-SMOKE DATASET CONVERSION COMPLETE")
    print("=" * 50)
    print(f"  Label mode:           {label_mode}")
    print(f"  Fire images:          {counts['fire']}")
    print(f"  NoFire images:        {counts['nofire']}")
    print(f"  Smoke-only (skipped): {counts['smoke_only_skipped']}")
    print(f"  Output: {output_root}")
    print("=" * 50)


if __name__ == "__main__":
    main()
