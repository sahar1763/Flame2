r"""
Convert_DFire_Dataset.py
------------------------
Converts the D-Fire dataset (YOLO format, classes: 0=smoke, 1=fire) into the
project's binary classification folder structure with unified bounding box labels.

Rules:
  - Image has any class=1 (fire) bbox  → Fire/   (includes fire+smoke images)
  - Image has only class=0 (smoke)     → SKIP    (ignored entirely)
  - Image has empty label file         → NoFire/

Output structure:
  D_Fire/
    Fire/
      images/       <- fire images copied here
      labels/       <- one label per image: single enclosing bbox (YOLO format, class=1)
    NoFire/
      images/       <- no-detection images copied here

Label format (YOLO): class_id x_center y_center width height
  - All bboxes in the image (fire AND smoke) are merged into one enclosing bbox.
  - The enclosing bbox is computed as: min(x1,y1) to max(x2,y2) across all boxes.

Label modes (--label_mode):
  - unified      -> merge fire + smoke boxes into one enclosing bbox (default).
  - biggest_fire -> keep only the single largest FIRE box (class 1; smoke filtered).
                    Output auto-suffixed with '_biggestfire' unless --output given.

Usage:
  # UNIFIED (default output = OUTPUT_ROOT)
  python Convert_DFire_Dataset.py --label_mode unified

  # BIGGEST FIRE (auto output = OUTPUT_ROOT + "_biggestfire")
  python Convert_DFire_Dataset.py --label_mode biggest_fire

  # Override raw source / output explicitly
  python Convert_DFire_Dataset.py --label_mode biggest_fire \
    --source "<raw_dfire_root>" \
    --output "J:\ControlProjects\FireDrone\Seperated_Dataset\D_Fire_biggestfire"
"""

import os
import shutil
import argparse

# === Configuration ===
SOURCE_ROOT = r"C:\Projects\Flame2\Seperated_Dataset\D_Fire_Org"
OUTPUT_ROOT = r"C:\Projects\Flame2\Seperated_Dataset\D_Fire"

SPLITS = ["train", "val", "test"]

# D-Fire classes: 0=smoke, 1=fire
FIRE_CLASS = 1
SMOKE_CLASS = 0

# Default labeling mode:
#   "unified"      -> merge all boxes into one enclosing bbox (fire + smoke)
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
            class_id = int(parts[0])
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
        description="Convert D-Fire dataset to binary classification structure.")
    parser.add_argument("--label_mode", choices=["unified", "biggest_fire"],
                        default=LABEL_MODE,
                        help="unified = merge all boxes into one enclosing bbox; "
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

    # Create output directories
    fire_images_dir = os.path.join(output_root, "Fire", "images")
    fire_labels_dir = os.path.join(output_root, "Fire", "labels")
    nofire_images_dir = os.path.join(output_root, "NoFire", "images")

    os.makedirs(fire_images_dir, exist_ok=True)
    os.makedirs(fire_labels_dir, exist_ok=True)
    os.makedirs(nofire_images_dir, exist_ok=True)

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

            if len(boxes) == 0:
                # No detections → NoFire
                shutil.copy2(img_path, os.path.join(nofire_images_dir, img_name))
                counts["nofire"] += 1

            else:
                # Check if any box is fire (class=1)
                has_fire = any(cls == FIRE_CLASS for cls, *_ in boxes)

                if has_fire:
                    # Fire image → copy image + create bbox label
                    shutil.copy2(img_path, os.path.join(fire_images_dir, img_name))

                    if label_mode == "unified":
                        # Merge ALL boxes (fire + smoke) into one enclosing bbox
                        enc_xc, enc_yc, enc_w, enc_h = compute_enclosing_bbox(boxes)
                    else:
                        # Keep only the largest fire bbox
                        enc_xc, enc_yc, enc_w, enc_h = select_biggest_fire_bbox(boxes, FIRE_CLASS)

                    # Write single-line label
                    out_label_path = os.path.join(fire_labels_dir, label_name)
                    with open(out_label_path, "w") as f:
                        f.write(f"{FIRE_CLASS} {enc_xc:.6f} {enc_yc:.6f} {enc_w:.6f} {enc_h:.6f}\n")

                    counts["fire"] += 1
                else:
                    # Smoke only → skip
                    counts["smoke_only_skipped"] += 1

    # --- Summary ---
    print("\n" + "=" * 50)
    print("D-FIRE DATASET CONVERSION COMPLETE")
    print("=" * 50)
    print(f"  Label mode:           {label_mode}")
    print(f"  Fire images:          {counts['fire']}")
    print(f"  NoFire images:        {counts['nofire']}")
    print(f"  Smoke-only (skipped): {counts['smoke_only_skipped']}")
    print(f"  Output: {output_root}")
    print("=" * 50)


if __name__ == "__main__":
    main()
