"""
compare_bbox_overlay.py
-----------------------
Visualize YOLO bounding boxes overlaid on D_Fire images.
Supports comparing original multi-class labels (D_Fire_Org) with merged single-class labels (D_Fire).

Usage:
  python compare_bbox_overlay.py --image WEB07375.jpg --save
  python compare_bbox_overlay.py --image WEB07375.jpg --dataset ..\Seperated_Dataset\D_Fire
  python compare_bbox_overlay.py --image WEB07375.jpg --org_labels ..\Seperated_Dataset\D_Fire_Org\Fire\labels
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_yolo_labels(label_path):
    """Parse all YOLO format bboxes from a label file."""
    if not os.path.exists(label_path):
        return []
    bboxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                bboxes.append((cls, xc, yc, w, h))
    return bboxes


def draw_yolo_bboxes(ax, bboxes, img_h, img_w, color_map=None, label_prefix=""):
    """Draw YOLO bboxes on a matplotlib axis."""
    if color_map is None:
        color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}

    for cls, xc, yc, w, h in bboxes:
        x_min = (xc - w / 2) * img_w
        y_min = (yc - h / 2) * img_h
        box_w = w * img_w
        box_h = h * img_h
        color = color_map.get(cls, 'yellow')
        rect = patches.Rectangle((x_min, y_min), box_w, box_h,
                                 linewidth=2, edgecolor=color, facecolor='none',
                                 linestyle='-')
        ax.add_patch(rect)
        label = f"{label_prefix}cls={cls}" if label_prefix else f"cls={cls}"
        ax.text(x_min, y_min - 4, label, color=color, fontsize=8,
                fontweight='bold', backgroundcolor='white')


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO bbox overlays on D_Fire images")
    parser.add_argument("--image", type=str, required=True,
                        help="Image filename (e.g. WEB07375.jpg)")
    parser.add_argument("--dataset", type=str, default=r"..\Seperated_Dataset\D_Fire",
                        help="Path to D_Fire dataset root")
    parser.add_argument("--org_labels", type=str, default=None,
                        help="Path to D_Fire_Org labels folder (for comparison)")
    parser.add_argument("--save", action="store_true",
                        help="Save figure instead of showing")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Output directory for saved figures")
    args = parser.parse_args()

    img_name = args.image
    stem = os.path.splitext(img_name)[0]

    # Find image
    img_path = os.path.join(args.dataset, "Fire", "images", img_name)
    if not os.path.exists(img_path):
        img_path = os.path.join(args.dataset, "NoFire", "images", img_name)
    if not os.path.exists(img_path):
        print(f"ERROR: Image '{img_name}' not found in dataset")
        return

    # Load image
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = image_rgb.shape[:2]

    # Load D_Fire (merged) labels
    label_path = os.path.join(args.dataset, "Fire", "labels", stem + ".txt")
    merged_bboxes = parse_yolo_labels(label_path)

    # Load D_Fire_Org (original multi-class) labels if available
    org_bboxes = []
    if args.org_labels:
        org_label_path = os.path.join(args.org_labels, stem + ".txt")
        org_bboxes = parse_yolo_labels(org_label_path)

    # Determine layout
    has_org = len(org_bboxes) > 0
    ncols = 2 if has_org else 1
    fig, axs = plt.subplots(1, ncols, figsize=(8 * ncols, 8))

    if ncols == 1:
        axs = [axs]

    # Panel 1: D_Fire merged labels
    axs[0].imshow(image_rgb)
    axs[0].set_title(f"D_Fire (merged) — {img_name}\n{len(merged_bboxes)} bbox(es)", fontsize=11)
    axs[0].axis('off')
    draw_yolo_bboxes(axs[0], merged_bboxes, img_h, img_w,
                     color_map={0: 'lime', 1: 'red'})

    # Panel 2: D_Fire_Org original labels (if provided)
    if has_org:
        axs[1].imshow(image_rgb)
        axs[1].set_title(f"D_Fire_Org (original) — {img_name}\n{len(org_bboxes)} bbox(es)", fontsize=11)
        axs[1].axis('off')
        # Original classes: 0=fire, 1=smoke
        draw_yolo_bboxes(axs[1], org_bboxes, img_h, img_w,
                         color_map={0: 'red', 1: 'cyan'})
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='fire (cls=0)'),
            Line2D([0], [0], color='cyan', lw=2, label='smoke (cls=1)'),
        ]
        axs[1].legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"bbox_overlay_{stem}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    # Print bbox info
    print(f"\nImage: {img_name} ({img_w}x{img_h})")
    print(f"D_Fire labels ({label_path}):")
    for cls, xc, yc, w, h in merged_bboxes:
        px_w, px_h = int(w * img_w), int(h * img_h)
        print(f"  cls={cls}  center=({xc:.4f}, {yc:.4f})  size=({w:.4f}, {h:.4f})  pixels=({px_w}x{px_h})")
    if org_bboxes:
        print(f"\nD_Fire_Org labels ({args.org_labels}):")
        for cls, xc, yc, w, h in org_bboxes:
            px_w, px_h = int(w * img_w), int(h * img_h)
            print(f"  cls={cls}  center=({xc:.4f}, {yc:.4f})  size=({w:.4f}, {h:.4f})  pixels=({px_w}x{px_h})")


if __name__ == "__main__":
    main()
