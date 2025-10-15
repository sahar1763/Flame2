import numpy as np
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
    x_min, y_min, x_max, y_max = bbox

    # Calculate bbox center
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # Get max side of the bbox
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    max_dim = max(bbox_width, bbox_height)
    if min_cropsize is not None:
        max_dim = max(max_dim, min_cropsize)

    # Final crop size
    crop_size = int(np.ceil(max_dim * crop_factor))
    half_crop = crop_size // 2

    # Compute crop bounds
    y1 = max(center_y - half_crop, 0)
    y2 = min(center_y + half_crop, image.shape[0])
    x1 = max(center_x - half_crop, 0)
    x2 = min(center_x + half_crop, image.shape[1])

    croppedImage = image[y1:y2, x1:x2, :]
    print(croppedImage.shape)

    # Crop and return
    return image[y1:y2, x1:x2, :]


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

    # === Cropped patches with predictions
    for i, (crop, pred, conf) in enumerate(zip(crops_np, predictions, confidences)):
        if crop.size == 0:
            axs[i + 1].axis('off')
            axs[i + 1].set_title(f"Crop {i + 1}\nEMPTY")
            continue
        axs[i + 1].imshow(crop)
        axs[i + 1].set_title(f"Crop {i + 1}\n{pred} ({conf:.2f})")
        axs[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


def predict_crops_majority_vote(crops, model, bbox, device,
                                original_image=None,
                                crops_np=None,
                                plot=False):
    label_names = {0: "No Fire", 1: "Fire"}

    times = {}

    t0 = time.perf_counter()
    model.eval()

    # Stage 1: Stack crops
    t1 = time.perf_counter()
    batch = torch.stack(crops)
    times['stack'] = time.perf_counter() - t1

    # Stage 2: Move to device
    t2 = time.perf_counter()
    batch = batch.to(device)
    times['to_device'] = time.perf_counter() - t2

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
    with torch.no_grad():
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
    final_class = 1 if vote_ratio > 0.6 else 0
    final_label = label_names[final_class]
    avg_conf = probs[:, final_class].mean().item()
    times['postprocess'] = time.perf_counter() - t4

    total_time = time.perf_counter() - t0

    # === Print timing breakdown ===
    print(f"\n=== Inference Timing Breakdown ===")
    for k, v in times.items():
        print(f"{k:>12}: {v * 1000:.2f} ms")
    print(f"{'Total':>12}: {total_time * 1000:.2f} ms\n")

    # Optional plot
    if plot and original_image is not None and crops_np is not None:
        plot_crops_with_predictions(
            original_image,
            crops_np,
            pred_labels,
            confidence_scores,
            final_label,
            avg_conf,
            bbox=bbox
        )

    return {
        "final_prediction": final_label,
        "confidence": avg_conf,
        "bbox": bbox
    } # TODO: Update the return based on ICD


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

    # Stage 1: Stack crops (Torch -> one batch tensor)
    t1 = time.perf_counter()
    batch = torch.stack(crops)  # (N,C,H,W)
    times['stack'] = time.perf_counter() - t1

    # Stage 2: Convert to NumPy (TensorRT expects np.float32)
    t2 = time.perf_counter()
    np_batch = batch.cpu().numpy().astype(np.float32)  # (N,C,H,W)
    times['to_numpy'] = time.perf_counter() - t2

    # Stage 3: Inference with TensorRT
    t3 = time.perf_counter()
    outputs = model.infer(np_batch)  # returns raw logits (N,num_classes)
    outputs = outputs.reshape(-1, 2) # TODO: Changed
    probs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)  # softmax
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    times['inference'] = time.perf_counter() - t3

    # Stage 4: Post-processing
    t4 = time.perf_counter()
    pred_labels = [label_names[p] for p in preds]
    fire_votes = (preds == 1).sum()
    vote_ratio = fire_votes / len(crops)
    final_class = 1 if vote_ratio > 0.6 else 0
    final_label = label_names[final_class]
    avg_conf = float(np.mean(probs[:, final_class]))
    times['postprocess'] = time.perf_counter() - t4

    total_time = time.perf_counter() - t0

    # === Print timing breakdown ===
    print(f"\n=== Inference Timing (TensorRT) ===")
    for k, v in times.items():
        print(f"{k:>12}: {v * 1000:.2f} ms")
    print(f"{'Total':>12}: {total_time * 1000:.2f} ms\n")

    # Optional plot
    if plot and original_image is not None and crops_np is not None:
        plot_crops_with_predictions(
            original_image,
            crops_np,
            pred_labels,
            confs.tolist(),
            final_label,
            avg_conf,
            bbox=bbox
        )

    return {
        "final_prediction": final_label,
        "confidence": avg_conf,
        "bbox": bbox
    }
