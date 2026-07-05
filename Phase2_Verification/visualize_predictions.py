"""
visualize_predictions.py
------------------------
Generate crop visualizations (like Figure 14) for specific images or all errors.

Modes:
  --errors    : Visualize all misclassified images (FP + FN) from a predictions.csv
  --images    : Visualize specific images by filename (comma-separated)
  --fp_only   : Only false positives
  --fn_only   : Only false negatives

Usage:
  python visualize_predictions.py --dataset ..\Seperated_Dataset\D_Fire --predictions benchmark_results\resnet18_pytorch\predictions.csv --errors
  python visualize_predictions.py --dataset ..\Seperated_Dataset\D_Fire --predictions benchmark_results\resnet18_pytorch\predictions.csv --images "WEB07375.jpg,WEB08123.jpg"
  python visualize_predictions.py --dataset ..\Seperated_Dataset\D_Fire --predictions benchmark_results\resnet18_pytorch\predictions.csv --fn_only
  python visualize_predictions.py --dataset ..\Seperated_Dataset\D_Fire --predictions benchmark_results\resnet18_pytorch\predictions.csv --fp_only
"""

import os
import sys
import argparse
import copy
import numpy as np
import cv2
import pandas as pd
import torch

sys.path.append(os.path.abspath(".."))

from wildfire_detector.utils_phase2_flow import plot_crops_with_predictions


# ===========================================================================
# Reuse helpers from benchmark_inference.py
# ===========================================================================

def parse_yolo_label(label_path):
    if not os.path.exists(label_path):
        return None
    with open(label_path, "r") as f:
        lines = f.readlines()
    if not lines:
        return None
    # Take the largest bbox by area
    best = None
    best_area = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            area = w * h
            if area > best_area:
                best_area = area
                best = (xc, yc, w, h)
    return best


def prepare_frame(image_rgb, canvas_h, canvas_w):
    H, W = image_rgb.shape[:2]
    scale = min(canvas_h / H, canvas_w / W)
    new_h = int(H * scale)
    new_w = int(W * scale)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    y_offset = (canvas_h - new_h) // 2
    x_offset = (canvas_w - new_w) // 2
    frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return frame, scale, y_offset, x_offset


def yolo_to_canvas_bbox(xc, yc, w, h, orig_H, orig_W, scale, y_offset, x_offset):
    new_h = int(orig_H * scale)
    new_w = int(orig_W * scale)
    c_min = int(round((xc - w / 2) * new_w))
    r_min = int(round((yc - h / 2) * new_h))
    c_max = int(round((xc + w / 2) * new_w))
    r_max = int(round((yc + h / 2) * new_h))
    r_min += y_offset
    r_max += y_offset
    c_min += x_offset
    c_max += x_offset
    return (r_min, c_min, r_max, c_max)


def compute_virtual_bbox(canvas_h, canvas_w, max_crop_factor):
    frame_min_dim = min(canvas_h, canvas_w)
    bbox_side = frame_min_dim / max_crop_factor
    center_r = canvas_h / 2.0
    center_c = canvas_w / 2.0
    half_side = bbox_side / 2.0
    r_min = int(round(center_r - half_side))
    c_min = int(round(center_c - half_side))
    r_max = int(round(center_r + half_side))
    c_max = int(round(center_c + half_side))
    return (r_min, c_min, r_max, c_max)


def build_metadata(dummy_md, bbox):
    md = copy.deepcopy(dummy_md)
    md["investigation_parameters"]["detected_bounding_box"] = [
        bbox[0], bbox[1], 0, bbox[2], bbox[3], 0
    ]
    return md


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize predictions with crop overlays")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to D_Fire dataset (Fire/NoFire structure)")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions.csv from benchmark run")
    parser.add_argument("--backend", type=str, default="pytorch",
                        choices=["pytorch", "tensorrt", "int8"])
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Output folder for saved plots")
    parser.add_argument("--errors", action="store_true",
                        help="Visualize all errors (FP + FN)")
    parser.add_argument("--fp_only", action="store_true",
                        help="Visualize only false positives")
    parser.add_argument("--fn_only", action="store_true",
                        help="Visualize only false negatives")
    parser.add_argument("--images", type=str, default=None,
                        help="Comma-separated list of image filenames to visualize")
    parser.add_argument("--max_images", type=int, default=50,
                        help="Maximum number of images to visualize (default: 50)")
    args = parser.parse_args()

    # Load predictions
    preds_df = pd.read_csv(args.predictions)
    print(f"Loaded {len(preds_df)} predictions from {args.predictions}")

    # Filter images to visualize
    if args.images:
        image_list = [x.strip() for x in args.images.split(",")]
        vis_df = preds_df[preds_df["image"].isin(image_list)]
    elif args.fp_only:
        vis_df = preds_df[(preds_df["pred_label"] == 1) & (preds_df["true_label"] == 0)]
        print(f"False Positives: {len(vis_df)}")
    elif args.fn_only:
        vis_df = preds_df[(preds_df["pred_label"] == 0) & (preds_df["true_label"] == 1)]
        print(f"False Negatives: {len(vis_df)}")
    elif args.errors:
        vis_df = preds_df[preds_df["pred_label"] != preds_df["true_label"]]
        print(f"Total errors: {len(vis_df)} (FP: {((vis_df['pred_label']==1) & (vis_df['true_label']==0)).sum()}, "
              f"FN: {((vis_df['pred_label']==0) & (vis_df['true_label']==1)).sum()})")
    else:
        print("No filter specified. Use --errors, --fp_only, --fn_only, or --images")
        return

    if vis_df.empty:
        print("No images match the filter.")
        return

    # Limit number
    if len(vis_df) > args.max_images:
        print(f"Limiting to {args.max_images} images (use --max_images to change)")
        vis_df = vis_df.head(args.max_images)

    # Initialize ScanManager
    if args.backend == "pytorch":
        from wildfire_detector.function_class_demo import ScanManager
    else:
        from wildfire_detector.function_class_demo_TensorRT import ScanManager

    print(f"Initializing ScanManager ({args.backend})...")
    sm = ScanManager(verbose=True)

    rgb_size = sm.config["image"]["rgb_size"]
    canvas_h, canvas_w = rgb_size[0], rgb_size[1]
    crop_factors = sm.config["phase2"]["crop_factors"]
    max_crop_factor = max(crop_factors)

    # Build dataset lookup
    fire_images_dir = os.path.join(args.dataset, "Fire", "images")
    fire_labels_dir = os.path.join(args.dataset, "Fire", "labels")
    nofire_images_dir = os.path.join(args.dataset, "NoFire", "images")

    # Create output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    fp_dir = os.path.join(args.output_dir, "false_positives")
    fn_dir = os.path.join(args.output_dir, "false_negatives")
    misc_dir = os.path.join(args.output_dir, "selected")
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)
    os.makedirs(misc_dir, exist_ok=True)

    print(f"\nGenerating visualizations for {len(vis_df)} images...\n")

    for idx, row in vis_df.iterrows():
        img_name = row["image"]
        true_label = int(row["true_label"])
        pred_label = int(row["pred_label"])

        # Find image path
        img_path = os.path.join(fire_images_dir, img_name)
        label_path = os.path.join(fire_labels_dir, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(img_path):
            img_path = os.path.join(nofire_images_dir, img_name)
            label_path = None

        if not os.path.exists(img_path):
            print(f"  SKIP: {img_name} not found")
            continue

        # Load and prepare
        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H_orig, W_orig = image_rgb.shape[:2]
        frame, scale, y_off, x_off = prepare_frame(image_rgb, canvas_h, canvas_w)

        # Determine bbox
        bbox = None
        if label_path and os.path.exists(label_path):
            parsed = parse_yolo_label(label_path)
            if parsed is not None:
                xc, yc, w, h = parsed
                bbox = yolo_to_canvas_bbox(xc, yc, w, h, H_orig, W_orig, scale, y_off, x_off)

        if bbox is None:
            bbox = compute_virtual_bbox(canvas_h, canvas_w, max_crop_factor)

        # Run phase2 with plot enabled
        md = build_metadata(sm.dummy_md, bbox)

        # Determine save path
        if pred_label == 1 and true_label == 0:
            error_type = "FP"
            save_dir = fp_dir
        elif pred_label == 0 and true_label == 1:
            error_type = "FN"
            save_dir = fn_dir
        else:
            error_type = "OK"
            save_dir = misc_dir

        save_name = f"{error_type}_{os.path.splitext(img_name)[0]}.png"
        save_path = os.path.join(save_dir, save_name)

        # Manual crop + classify to get per-crop results for plotting
        from wildfire_detector.utils_phase2_flow import crop_bbox_scaled, predict_crops_majority_vote
        image_size = sm.config["phase2"]["net_image_size"]
        bbox_pixels = bbox  # already in (r_min, c_min, r_max, c_max) format

        cropped_images_np = []
        for cf in crop_factors:
            cropped_np = crop_bbox_scaled(frame, bbox_pixels, cf, min_cropsize=image_size)
            cropped_images_np.append(cropped_np)

        # Resize and normalize
        resized_tensors = []
        for cropped_np in cropped_images_np:
            t = torch.from_numpy(cropped_np).permute(2, 0, 1).float()
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(image_size, image_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            resized_tensors.append(t)

        test_tensors = torch.stack(resized_tensors)
        test_tensors.div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        test_tensors = (test_tensors - mean) / std

        # Run predict (no plot internally) then plot manually with custom save_path
        final_label, avg_conf, _ = predict_crops_majority_vote(
            crops=test_tensors,
            model=sm.model,
            bbox=bbox_pixels,
            device=sm.device,
            original_image=frame,
            crops_np=cropped_images_np,
            plot=False,
            verbose=False,
            crop_factors=crop_factors
        )

        # Get per-crop predictions for the plot
        with torch.no_grad():
            batch = test_tensors.to(sm.device)
            outputs = sm.model(batch)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

        label_names = {0: "No Fire", 1: "Fire"}
        pred_labels_str = [label_names[p.item()] for p in preds]
        confidence_scores = probs.max(dim=1).values.cpu().numpy().tolist()
        final_label_str = label_names[final_label]

        # Convert bbox to (x_min, y_min, x_max, y_max) for plot
        r_min, c_min, r_max, c_max = bbox_pixels
        plot_bbox = (c_min, r_min, c_max, r_max)

        plot_crops_with_predictions(
            frame, cropped_images_np, pred_labels_str, confidence_scores,
            final_label_str, avg_conf,
            bbox=plot_bbox, crop_factors=crop_factors, save_path=save_path
        )

        conf_str = f"{row['confidence']:.2f}" if 'confidence' in row else ""
        print(f"  [{error_type}] {img_name} (true={true_label}, pred={pred_label}, conf={conf_str}) → {save_path}")

    print(f"\nDone! Visualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
