r"""
Convert_YOLO_Labels_To_Unified_BBox.py
--------------------------------------
Reads all YOLO label files from an input folder and creates a new label folder.

Label modes (--label_mode):
- unified      -> merge ALL boxes in a file into one enclosing box (default).
- biggest_fire -> keep only the single largest box (by area). IAI is fire-only.

For every label file:
- Empty file -> empty output file.
- Original classes are ignored; the output box is always written with class ID 1.

YOLO format:
class_id x_center y_center width height

Input: a folder of .txt YOLO label files (searched recursively). Images are NOT
touched. For IAI, point --input at the raw multi-box label folder.

Usage:
  # UNIFIED (merge all boxes) -> new output folder (existing labels untouched)
  python Convert_YOLO_Labels_To_Unified_BBox.py --label_mode unified \
    --input  "J:\ControlProjects\FireDrone\Seperated_Dataset\IAI_Datasets\Fire\IAI_raw_Label\IAI_raw_Label\labels" \
    --output "J:\ControlProjects\FireDrone\Seperated_Dataset\IAI_Datasets\Fire\labels_unified"

  # BIGGEST FIRE (largest box only)
  python Convert_YOLO_Labels_To_Unified_BBox.py --label_mode biggest_fire \
    --input  "J:\ControlProjects\FireDrone\Seperated_Dataset\IAI_Datasets\Fire\IAI_raw_Label\IAI_raw_Label\labels" \
    --output "J:\ControlProjects\FireDrone\Seperated_Dataset\IAI_Datasets\Fire\labels_biggestfire"
"""

import argparse
from pathlib import Path


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

INPUT_LABELS_FOLDER = Path(r"C:\Projects\Flame2\Seperated_Dataset\IAI_Datasets\Fire\labels")
OUTPUT_LABELS_FOLDER = Path(r"C:\Projects\Flame2\Seperated_Dataset\IAI_Datasets\Fire\labels_new")

OUTPUT_CLASS_ID = 1
DECIMAL_PLACES = 6

# Default labeling mode:
#   "unified"      -> merge all boxes into one enclosing bbox
#   "biggest_fire" -> keep only the single largest box (IAI has fire only)
LABEL_MODE = "unified"


def parse_yolo_file(label_path: Path):
    """
    Read a YOLO annotation file.

    Returns:
        List of tuples:
        [(x_center, y_center, width, height), ...]
    """
    boxes = []

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO annotation in {label_path}, "
                    f"line {line_number}: expected 5 values, got {len(parts)}."
                )

            try:
                # The original class is intentionally ignored.
                _, x_center, y_center, width, height = map(float, parts)
            except ValueError as error:
                raise ValueError(
                    f"Invalid numeric value in {label_path}, "
                    f"line {line_number}: {line}"
                ) from error

            boxes.append((x_center, y_center, width, height))

    return boxes


def compute_enclosing_bbox(boxes):
    """
    Merge all normalized YOLO boxes into one enclosing normalized YOLO box.

    Args:
        boxes:
            [(x_center, y_center, width, height), ...]

    Returns:
        (x_center, y_center, width, height)
    """
    x_min = min(x_center - width / 2.0 for x_center, _, width, _ in boxes)
    y_min = min(y_center - height / 2.0 for _, y_center, _, height in boxes)
    x_max = max(x_center + width / 2.0 for x_center, _, width, _ in boxes)
    y_max = max(y_center + height / 2.0 for _, y_center, _, height in boxes)

    # Keep the merged box inside normalized image coordinates.
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    x_max = min(1.0, x_max)
    y_max = min(1.0, y_max)

    width = x_max - x_min
    height = y_max - y_min
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    return x_center, y_center, width, height


def select_biggest_box(boxes):
    """
    Return the single largest box (by normalized area) as
    (x_center, y_center, width, height). IAI contains only fire boxes.
    """
    x_center, y_center, width, height = max(boxes, key=lambda b: b[2] * b[3])
    return x_center, y_center, width, height


def process_label_file(input_path: Path, output_path: Path, label_mode="unified"):
    boxes = parse_yolo_file(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not boxes:
        # Preserve negative samples as empty YOLO label files.
        output_path.write_text("", encoding="utf-8")
        return "empty"

    if label_mode == "unified":
        # Merge all boxes into one enclosing bbox.
        x_center, y_center, width, height = compute_enclosing_bbox(boxes)
    else:
        # IAI contains only fire boxes -> keep the single largest one.
        x_center, y_center, width, height = select_biggest_box(boxes)

    output_line = (
        f"{OUTPUT_CLASS_ID} "
        f"{x_center:.{DECIMAL_PLACES}f} "
        f"{y_center:.{DECIMAL_PLACES}f} "
        f"{width:.{DECIMAL_PLACES}f} "
        f"{height:.{DECIMAL_PLACES}f}\n"
    )

    output_path.write_text(output_line, encoding="utf-8")
    return "converted"


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO labels to a single bbox per image (IAI: fire only).")
    parser.add_argument("--label_mode", choices=["unified", "biggest_fire"],
                        default=LABEL_MODE,
                        help="unified = merge all boxes into one enclosing bbox; "
                             "biggest_fire = keep only the largest fire bbox.")
    parser.add_argument("--input", default=None, help="Input labels folder.")
    parser.add_argument("--output", default=None,
                        help="Output labels folder. Default: OUTPUT_LABELS_FOLDER, "
                             "with '_biggest' suffix in biggest_fire mode.")
    args = parser.parse_args()

    label_mode = args.label_mode
    input_folder = Path(args.input) if args.input else INPUT_LABELS_FOLDER
    if args.output:
        output_folder = Path(args.output)
    elif label_mode == "unified":
        output_folder = OUTPUT_LABELS_FOLDER
    else:
        output_folder = OUTPUT_LABELS_FOLDER.parent / (OUTPUT_LABELS_FOLDER.name + "_biggest")

    print(f"Label mode: {label_mode}")

    if not input_folder.is_dir():
        raise FileNotFoundError(
            f"Input labels folder does not exist:\n{input_folder}"
        )

    if input_folder.resolve() == output_folder.resolve():
        raise ValueError(
            "The input and output folders must be different "
            "to avoid overwriting the original annotations."
        )

    label_files = sorted(input_folder.rglob("*.txt"))

    if not label_files:
        raise RuntimeError(
            f"No .txt label files were found in:\n{input_folder}"
        )

    converted_count = 0
    empty_count = 0
    failed_count = 0

    for input_path in label_files:
        relative_path = input_path.relative_to(input_folder)
        output_path = output_folder / relative_path

        try:
            result = process_label_file(input_path, output_path, label_mode)

            if result == "converted":
                converted_count += 1
            else:
                empty_count += 1

        except Exception as error:
            failed_count += 1
            print(f"ERROR: {input_path}")
            print(f"       {error}")

    print("\n" + "=" * 55)
    print("LABEL CONVERSION COMPLETE")
    print("=" * 55)
    print(f"Label mode:                  {label_mode}")
    print(f"Non-empty labels converted: {converted_count}")
    print(f"Empty labels preserved:      {empty_count}")
    print(f"Failed files:                {failed_count}")
    print(f"Output folder:               {output_folder}")
    print("=" * 55)


if __name__ == "__main__":
    main()
