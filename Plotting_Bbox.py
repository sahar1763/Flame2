from __future__ import annotations

import re
from pathlib import Path

import cv2


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

IMAGES_FOLDER = Path(
    r"C:\Projects\Flame2\Seperated_Dataset\D_Fire\Fire\images"
)

LABELS_FOLDER = Path(
    r"C:\Projects\Flame2\Seperated_Dataset\D_Fire\Fire\labels_unified"
)

DISPLAY_FPS = 10.0

# Optional output video. Keep None to display only.
OUTPUT_VIDEO: Path | None = None

# Example:
# OUTPUT_VIDEO = Path(r"C:\path\to\annotated_sequence.mp4")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def natural_sort_key(path: Path) -> list:
    """Sort img_2 before img_10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def load_image_paths(images_folder: Path) -> list[Path]:
    """Load all supported images from the images folder."""
    if not images_folder.is_dir():
        raise FileNotFoundError(
            f"Images folder does not exist:\n{images_folder}"
        )

    image_paths = [
        path
        for path in images_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_paths.sort(key=natural_sort_key)
    return image_paths


def get_label_path(
    image_path: Path,
    labels_folder: Path,
) -> Path:
    """
    Match image and label by filename stem.

    Example:
        images/img_000017.jpg
        labels/img_000017.txt
    """
    return labels_folder / f"{image_path.stem}.txt"


# ---------------------------------------------------------------------
# YOLO annotation functions
# ---------------------------------------------------------------------

def read_yolo_annotations(
    annotation_path: Path,
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int, int]]:
    """
    Read YOLO detection annotations.

    Expected format:
        class_id x_center y_center width height

    Coordinates are normalized to [0, 1].
    """
    boxes = []

    if not annotation_path.exists():
        return boxes

    lines = annotation_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        values = line.split()

        if len(values) != 5:
            print(
                f"Warning: invalid annotation in "
                f"{annotation_path.name}, line {line_number}: {line}"
            )
            continue

        try:
            class_id = int(float(values[0]))
            center_x = float(values[1])
            center_y = float(values[2])
            box_width = float(values[3])
            box_height = float(values[4])

        except ValueError:
            print(
                f"Warning: non-numeric annotation in "
                f"{annotation_path.name}, line {line_number}: {line}"
            )
            continue

        x1 = round(
            (center_x - box_width / 2.0) * image_width
        )
        y1 = round(
            (center_y - box_height / 2.0) * image_height
        )
        x2 = round(
            (center_x + box_width / 2.0) * image_width
        )
        y2 = round(
            (center_y + box_height / 2.0) * image_height
        )

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        if x2 <= x1 or y2 <= y1:
            print(
                f"Warning: invalid bounding box in "
                f"{annotation_path.name}, line {line_number}"
            )
            continue

        boxes.append(
            (class_id, x1, y1, x2, y2)
        )

    return boxes


def draw_boxes(
    image,
    boxes: list[tuple[int, int, int, int, int]],
) -> None:
    """Draw bounding boxes."""
    for class_id, x1, y1, x2, y2 in boxes:
        CLASS_NAMES = {
            0: "Smoke",
            1: "Fire",
        }

        CLASS_COLORS = {
            0: (255, 0, 0),  # Blue: Smoke
            1: (0, 0, 255),  # Red: Fire
        }

        label = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}",
        )

        box_color = CLASS_COLORS.get(
            class_id,
            (0, 255, 0),  # Green for unknown classes
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            box_color,
            2,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            font_thickness,
        )

        label_height = text_height + baseline + 8

        if y1 >= label_height:
            label_top = y1 - label_height
            label_bottom = y1
            text_y = y1 - baseline - 4
        else:
            label_top = y1
            label_bottom = min(
                image.shape[0] - 1,
                y1 + label_height,
            )
            text_y = min(
                image.shape[0] - baseline - 1,
                y1 + text_height + 4,
            )

        cv2.rectangle(
            image,
            (x1, label_top),
            (
                min(
                    image.shape[1] - 1,
                    x1 + text_width + 8,
                ),
                label_bottom,
            ),
            box_color,
            -1,
        )

        cv2.putText(
            image,
            label,
            (x1 + 4, text_y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )


def add_image_information(
    image,
    image_index: int,
    total_images: int,
    image_name: str,
    number_of_boxes: int,
    label_exists: bool,
) -> None:
    status = "label found" if label_exists else "no label file"

    text = (
        f"Image {image_index + 1}/{total_images} | "
        f"{image_name} | Boxes: {number_of_boxes} | "
        f"{status}"
    )

    cv2.putText(
        image,
        text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    if DISPLAY_FPS <= 0:
        raise ValueError(
            "DISPLAY_FPS must be greater than zero."
        )

    if not LABELS_FOLDER.is_dir():
        raise FileNotFoundError(
            f"Labels folder does not exist:\n{LABELS_FOLDER}"
        )

    image_paths = load_image_paths(IMAGES_FOLDER)

    if not image_paths:
        raise RuntimeError(
            f"No supported image files were found in:\n{IMAGES_FOLDER}"
        )

    label_paths = [
        get_label_path(
            image_path=image_path,
            labels_folder=LABELS_FOLDER,
        )
        for image_path in image_paths
    ]

    number_with_labels = sum(
        label_path.exists()
        for label_path in label_paths
    )

    print(f"Images folder: {IMAGES_FOLDER}")
    print(f"Labels folder: {LABELS_FOLDER}")
    print(f"Images found: {len(image_paths)}")
    print(
        f"Images with label files: "
        f"{number_with_labels}/{len(image_paths)}"
    )

    delay_ms = max(
        1,
        round(1000 / DISPLAY_FPS),
    )

    image_index = 0
    paused = False

    writer = None
    output_size = None
    written_indices = set()

    print()
    print("Controls:")
    print("  Space: pause/resume")
    print("  D or Right Arrow: next image")
    print("  A or Left Arrow: previous image")
    print("  Q or Esc: quit")

    while 0 <= image_index < len(image_paths):
        image_path = image_paths[image_index]
        annotation_path = label_paths[image_index]

        image = cv2.imread(str(image_path))

        if image is None:
            print(
                f"Warning: could not read image: {image_path}"
            )
            image_index += 1
            continue

        image_height, image_width = image.shape[:2]

        boxes = read_yolo_annotations(
            annotation_path=annotation_path,
            image_width=image_width,
            image_height=image_height,
        )

        draw_boxes(
            image=image,
            boxes=boxes,
        )

        add_image_information(
            image=image,
            image_index=image_index,
            total_images=len(image_paths),
            image_name=image_path.name,
            number_of_boxes=len(boxes),
            label_exists=annotation_path.exists(),
        )

        if OUTPUT_VIDEO is not None and writer is None:
            OUTPUT_VIDEO.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_size = (
                image_width,
                image_height,
            )

            writer = cv2.VideoWriter(
                str(OUTPUT_VIDEO),
                cv2.VideoWriter_fourcc(*"mp4v"),
                DISPLAY_FPS,
                output_size,
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Could not create output video:\n{OUTPUT_VIDEO}"
                )

        if writer is not None and output_size is not None:
            output_image = image

            if (
                image_width,
                image_height,
            ) != output_size:
                output_image = cv2.resize(
                    image,
                    output_size,
                    interpolation=cv2.INTER_AREA,
                )

            if image_index not in written_indices:
                writer.write(output_image)
                written_indices.add(image_index)

        cv2.imshow(
            "Unified YOLO Bounding Boxes",
            image,
        )

        key = cv2.waitKey(
            0 if paused else delay_ms
        ) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord(" "):
            paused = not paused
            continue

        if key in (ord("d"), 83):
            image_index = min(
                image_index + 1,
                len(image_paths) - 1,
            )
            continue

        if key in (ord("a"), 81):
            image_index = max(
                image_index - 1,
                0,
            )
            continue

        if not paused:
            image_index += 1

    if writer is not None:
        writer.release()
        print(
            f"Output video saved to:\n{OUTPUT_VIDEO}"
        )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()