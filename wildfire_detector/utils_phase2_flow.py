import numpy as np
import os
import time
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def crop_bbox_scaled(image, bbox, crop_factor, min_cropsize=None):
    """
    Crop an RGB image around a bounding box, enlarging it by a crop factor.

    Parameters:
        image (np.ndarray): RGB image, shape (H, W, 3)
        bbox (tuple): (x_min, y_min, x_max, y_max)
        crop_factor (float): Scaling factor for the crop size

    Returns:
        np.ndarray: Cropped RGB image
    """
    H, W = image.shape[:2]
    r_min, c_min, r_max, c_max = bbox

    # --- pixel geometry ---
    bbox_height = r_max - r_min
    bbox_width = c_max - c_min
    max_dim = max(bbox_height, bbox_width, 1)

    if min_cropsize is not None:
        max_dim = max(max_dim, min_cropsize)

    crop_size = int(np.ceil(max_dim * crop_factor))
    crop_size = max(crop_size, 1)
    half = crop_size / 2.0

    # --- true geometric center (float) ---
    center_r = (r_min + r_max) / 2.0
    center_c = (c_min + c_max) / 2.0

    r1 = int(round(center_r - half))
    c1 = int(round(center_c - half))
    r2 = r1 + crop_size
    c2 = c1 + crop_size

    # --- shift window if outside image (DON’T SHRINK) ---
    if r1 < 0:
        r2 -= r1
        r1 = 0
    if c1 < 0:
        c2 -= c1
        c1 = 0
    if r2 > H:
        r1 -= (r2 - H)
        r2 = H
    if c2 > W:
        c1 -= (c2 - W)
        c2 = W

    r1 = max(r1, 0)
    c1 = max(c1, 0)

    croppedImage = image[r1:r2, c1:c2, :]

    return croppedImage


def plot_crops_with_predictions(original_image, crops_np, predictions, confidences, final_pred, final_conf, bbox=None, crop_factors=None, save_path=None, min_cropsize=224):
    """
    Displays the original image with bbox and each crop with predicted label and confidence.

    Parameters:
        original_image (np.ndarray): RGB image
        crops_np (List[np.ndarray]): List of cropped RGB images
        predictions (List[str]): Predicted label per crop
        confidences (List[float]): Confidence score per crop
        final_pred (str): Aggregated prediction
        final_conf (float): Aggregated confidence
        bbox (tuple): (x_min, y_min, x_max, y_max) - optional
        crop_factors (List[float]): Scale factors for each crop (for labeling)
        save_path (str): Full path to save figure (default: results_demoPackage/crops_phase2.png)
        min_cropsize (int): Minimum crop size used in crop_bbox_scaled (default: 224)
    """
    #TODO: Verify changes of this function and check if using the function class instead
    num_crops = len(crops_np)
    fig, axs = plt.subplots(1, num_crops + 1, figsize=(5 * (num_crops + 1), 5))

    # Handle single-crop case (axs not iterable)
    if num_crops == 1:
        axs = [axs] if not hasattr(axs, '__len__') else axs

    # === Original image with optional bbox
    axs[0].imshow(original_image)
    axs[0].set_title(f"Original Image\nFinal: {final_pred} ({final_conf:.2f})", fontsize=11)
    axs[0].axis('off')

    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=2, edgecolor='red', facecolor='none')
        axs[0].add_patch(rect)

    # === Cropped patches with predictions and colored borders
    # To draw the bbox correctly inside each crop, we replicate the crop_bbox_scaled
    # shift logic to find where the bbox actually falls inside the crop window.
    # bbox is (x_min, y_min, x_max, y_max) here
    if bbox is not None:
        x_min_b, y_min_b, x_max_b, y_max_b = bbox
        bbox_w = x_max_b - x_min_b
        bbox_h = y_max_b - y_min_b
        # In crop_bbox_scaled, bbox is (r_min, c_min, r_max, c_max) = (y_min, x_min, y_max, x_max)
        r_min_b, c_min_b, r_max_b, c_max_b = y_min_b, x_min_b, y_max_b, x_max_b
        bbox_height = r_max_b - r_min_b
        bbox_width = c_max_b - c_min_b
        max_dim = max(bbox_height, bbox_width, 1)
        max_dim = max(max_dim, min_cropsize)
    else:
        bbox_w = bbox_h = 0

    # Get original image dimensions (needed for shift computation)
    img_H, img_W = original_image.shape[:2]

    for i, (crop, pred, conf) in enumerate(zip(crops_np, predictions, confidences)):
        ax = axs[i + 1]
        if crop.size == 0:
            ax.axis('off')
            ax.set_title(f"Crop {i + 1}\nEMPTY")
            continue
        crop_h, crop_w = crop.shape[:2]
        ax.imshow(crop)

        # Draw bbox inside crop — replicate crop_bbox_scaled shift logic
        if bbox is not None and crop_factors and i < len(crop_factors):
            cf = crop_factors[i]
            crop_size = int(np.ceil(max_dim * cf))
            crop_size = max(crop_size, 1)
            half = crop_size / 2.0

            # True geometric center of bbox
            center_r = (r_min_b + r_max_b) / 2.0
            center_c = (c_min_b + c_max_b) / 2.0

            # Compute crop window (same as crop_bbox_scaled)
            r1 = int(round(center_r - half))
            c1 = int(round(center_c - half))
            r2 = r1 + crop_size
            c2 = c1 + crop_size

            # Shift if outside image
            if r1 < 0:
                r2 -= r1
                r1 = 0
            if c1 < 0:
                c2 -= c1
                c1 = 0
            if r2 > img_H:
                r1 -= (r2 - img_H)
                r2 = img_H
            if c2 > img_W:
                c1 -= (c2 - img_W)
                c2 = img_W
            r1 = max(r1, 0)
            c1 = max(c1, 0)

            # Actual crop dimensions (may differ from crop_size at edges)
            actual_crop_w = c2 - c1
            actual_crop_h = r2 - r1

            # Bbox position relative to crop window origin
            rel_x = (c_min_b - c1) / actual_crop_w * crop_w
            rel_y = (r_min_b - r1) / actual_crop_h * crop_h
            rect_w = bbox_width / actual_crop_w * crop_w
            rect_h = bbox_height / actual_crop_h * crop_h

            rect = patches.Rectangle((rel_x, rel_y), rect_w, rect_h,
                                     linewidth=2, edgecolor='red', facecolor='none',
                                     linestyle='--')
            ax.add_patch(rect)

        # Title with scale factor
        scale_str = f" (×{crop_factors[i]:.2f})" if crop_factors and i < len(crop_factors) else ""
        ax.set_title(f"Crop {i + 1}{scale_str}\n{pred} ({conf:.2f})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    plt.tight_layout()

    # Save
    if save_path is None:
        save_dir = "results_demoPackage"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "crops_phase2.png")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved crop visualization: {save_path}")


def predict_crops_majority_vote(crops, model, bbox, device,
                                original_image=None,
                                crops_np=None,
                                plot=False,
                                verbose=True,
                                crop_factors=None):

    label_names = {0: "No Fire", 1: "Fire"}
    times = {}
    t0 = time.perf_counter()
    model.eval()

    # Protection if there are no crops
    if (isinstance(crops, list) and len(crops) == 0) or (torch.is_tensor(crops) and crops.nelement() == 0):
        return "No Fire", 0.0

    # Prepare Batch and Move to Device
    t1 = time.perf_counter()

    if isinstance(crops, list):
        # Old way: list of tensors
        batch = torch.stack(crops).to(device)
    else:
        batch = crops.to(device)

    times['prepare_batch'] = time.perf_counter() - t1

    # # Stage 3: Inference
    # t3 = time.time()
    # with torch.no_grad():
    #     outputs = model(batch)
    #     probs = F.softmax(outputs, dim=1)
    #     preds = torch.argmax(probs, dim=1)
    # times['inference'] = time.time() - t3

    # Stage 3: Inference + softmax + argmax (everything GPU-related)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    with torch.inference_mode(): # TODO: Originally "with torch.no_grad():"
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times['inference'] = time.perf_counter() - t3

    # Stage 4: Post-processing
    t4 = time.perf_counter()
    pred_labels = [label_names[p.item()] for p in preds]
    confidence_scores = probs.max(dim=1).values.cpu().numpy().tolist()
    fire_votes = (preds == 1).sum().item()
    vote_ratio = fire_votes / len(crops)
    final_class = 1 if vote_ratio >= 0.5 else 0
    final_label = label_names[final_class]
    # avg_conf = probs[:, final_class].mean().item()

    preds_np = preds.cpu().numpy()
    probs_np = probs.cpu().numpy()

    # confidence of each prediction (max softmax)
    pred_conf = probs_np[np.arange(len(preds_np)), preds_np]

    # agreement-aware confidence
    conf_effective = np.where(
        preds_np == final_class,
        pred_conf,
        1.0 - pred_conf
    )

    avg_conf = float(conf_effective.mean())

    times['postprocess'] = time.perf_counter() - t4

    total_time = time.perf_counter() - t0

    # === Print timing breakdown ===
    if verbose:
        print(f"\n=== Inference Timing Breakdown ===")
        for k, v in times.items():
            print(f"{k:>12}: {v * 1000:.2f} ms")
        print(f"{'Total':>12}: {total_time * 1000:.2f} ms\n")

    # Optional plot
    if plot and original_image is not None and crops_np is not None:
        y_min, x_min, y_max, x_max = bbox
        plot_crops_with_predictions(
            original_image,
            crops_np,
            pred_labels,
            confidence_scores,
            final_label,
            avg_conf,
            bbox=(x_min, y_min, x_max, y_max),
            crop_factors=crop_factors
        )

    return final_class, avg_conf, times


def predict_crops_majority_vote_RT(crops, model, bbox,
                                original_image=None,
                                crops_np=None,
                                plot=False,
                                verbose=True,
                                crop_factors=None):
    """
    TensorRT-only version (batched), similar to original PyTorch one.
    'model' here is a TRTInference instance.
    """
    label_names = {0: "No Fire", 1: "Fire"}
    times = {}
    t0 = time.perf_counter()

    # Protection if there are no crops
    if (isinstance(crops, list) and len(crops) == 0) or (torch.is_tensor(crops) and crops.nelement() == 0):
        return "No Fire", 0.0

    # Stage 1: Handle Batch
    t1 = time.perf_counter()
    if isinstance(crops, list):
        batch = torch.stack(crops)
    else:
        batch = crops  # It's already a stacked tensor!
    times['stack'] = time.perf_counter() - t1

    # Stage 2: Convert to NumPy (TensorRT expects np.float32)
    t2 = time.perf_counter()
    np_batch = batch.detach().cpu().numpy().astype(np.float32)  # (N,C,H,W)
    times['to_numpy'] = time.perf_counter() - t2

    # Stage 3: Inference with TensorRT
    t3 = time.perf_counter()
    outputs = model.infer(np_batch)  # returns raw logits (N,num_classes)
    outputs = outputs.reshape(-1, 2).astype(np.float32)  # e.g., num_classes = 2
    exp_logits = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
    probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-12) # Softmax
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    times['inference'] = time.perf_counter() - t3

    # Stage 4: Post-processing
    t4 = time.perf_counter()
    pred_labels = [label_names[p] for p in preds]
    fire_votes = (preds == 1).sum()
    vote_ratio = fire_votes / len(crops)
    final_class = 1 if vote_ratio >= 0.5 else 0
    final_label = label_names[final_class]
    # avg_conf = float(np.mean(probs[:, final_class]))

    pred_conf = probs[np.arange(len(preds)), preds]

    conf_effective = np.where(
        preds == final_class,
        pred_conf,
        1.0 - pred_conf
    )

    avg_conf = float(conf_effective.mean())

    times['postprocess'] = time.perf_counter() - t4

    total_time = time.perf_counter() - t0

    # === Print timing breakdown ===
    if verbose:
        print(f"\n=== Inference Timing (TensorRT) ===")
        for k, v in times.items():
            print(f"{k:>12}: {v * 1000:.2f} ms")
        print(f"{'Total':>12}: {total_time * 1000:.2f} ms\n")

    # Optional plot
    if plot and original_image is not None and crops_np is not None:
        y_min, x_min, y_max, x_max = bbox
        plot_crops_with_predictions(
            original_image,
            crops_np,
            pred_labels,
            confs.tolist(),
            final_label,
            avg_conf,
            bbox=(x_min, y_min, x_max, y_max),
            crop_factors=crop_factors
        )

    return final_class, avg_conf, times
