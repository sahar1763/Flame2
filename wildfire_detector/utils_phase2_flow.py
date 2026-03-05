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


def plot_crops_with_predictions(original_image, crops_np, predictions, confidences, final_pred, final_conf, bbox=None):
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
    """
    num_crops = len(crops_np)
    fig, axs = plt.subplots(1, num_crops + 1, figsize=(5 * (num_crops + 1), 5))

    # === Original image with optional bbox
    axs[0].imshow(original_image)
    axs[0].set_title(f"Original Image\nFinal: {final_pred} ({final_conf:.2f})")
    axs[0].axis('off')

    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=2, edgecolor='red', facecolor='none')
        axs[0].add_patch(rect)

    save_dir = "results_demoPackage"
    filename = "crops_phase2.png"
    os.makedirs(save_dir, exist_ok=True)
    # === Cropped patches with predictions
    for i, (crop, pred, conf) in enumerate(zip(crops_np, predictions, confidences)):
        ax = axs[i + 1]
        if crop.size == 0:
            ax.axis('off')
            ax.set_title(f"Crop {i + 1}\nEMPTY")
            continue
        ax.imshow(crop)
        ax.set_title(f"Crop {i + 1}\n{pred} ({conf:.2f})")
        ax.axis('off')

    plt.tight_layout()

    # Save instead of show
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def predict_crops_majority_vote(crops, model, bbox, device,
                                original_image=None,
                                crops_np=None,
                                plot=False):

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
            bbox=(x_min, y_min, x_max, y_max)
        )

    return final_class, avg_conf


def predict_crops_majority_vote_RT(crops, model, bbox,
                                original_image=None,
                                crops_np=None,
                                plot=False):
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
            bbox=(x_min, y_min, x_max, y_max)
        )

    return final_class, avg_conf
