import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import time


# ============================================================
# DN → Temperature Conversion (calibrated IR camera)
# ============================================================
# T = (1.0893 + sqrt(0.0296*DN - 2.349)) / 0.0148
# For DN < 80: T = 82.9 °C (domain constraint: sqrt argument < 0)

def _build_dn_to_temperature_lut(dn_max=255):
    """
    Pre-compute a lookup table mapping DN values [0..dn_max] to temperature [°C].
    Uses the calibration equation. For DN < 80, temperature is clamped to 82.9°C.
    """
    dn = np.arange(dn_max + 1, dtype=np.float32)
    sqrt_arg = 0.0296 * dn - 2.349
    # Clip negative values (DN < ~79.4) to 0 before sqrt, then override with constant
    sqrt_arg_safe = np.maximum(sqrt_arg, 0.0)
    temperature = (1.0893 + np.sqrt(sqrt_arg_safe)) / 0.0148
    # Clamp DN < 80 to fixed temperature
    temperature[dn < 80] = 82.9
    return temperature


# Global LUT — built once at import, zero-cost per-frame conversion
DN_TO_TEMP_LUT = _build_dn_to_temperature_lut(dn_max=255)


def dn_to_temperature(frame):
    """
    Convert a raw IR frame (uint8 DN) to temperature [°C] using a pre-built LUT.
    Single array-index operation — O(1) per pixel, no per-pixel math.

    Parameters
    ----------
    frame : ndarray, dtype uint8, shape (H, W)
        Raw digital number image from the IR sensor.

    Returns
    -------
    ndarray, dtype float32, shape (H, W)
        Temperature image in degrees Celsius.
    """
    return DN_TO_TEMP_LUT[frame]


def create_homography(src_yx, dst_yx):
    """
    Compute a homography from (y, x) source points to (y, x) destination points.
    Internally converts to OpenCV (x, y) convention so the returned H is
    compatible with cv2.warpPerspective.

    Parameters:
        src_yx: ndarray (N, 2) — source points in (y, x) order
        dst_yx: ndarray (N, 2) — destination points in (y, x) order

    Returns:
        H: 3x3 homography matrix in (x, y) convention
        mask: inlier mask from cv2.findHomography
    """
    src_xy = src_yx[:, ::-1]
    dst_xy = dst_yx[:, ::-1]
    H, mask = cv2.findHomography(src_xy, dst_xy)
    return H, mask


def project_points_with_homography(points_yx, H):
    """
    Project (y, x) points through a homography H (in OpenCV x,y convention).
    Input and output are in (y, x) order.

    Parameters:
        points_yx: ndarray (N, 2) — points in (y, x) order
        H: 3x3 homography matrix in (x, y) convention

    Returns:
        ndarray (N, 2) — projected points in (y, x) order
    """
    # Flip (y, x) → (x, y) for homography math
    points_xy = points_yx[:, ::-1]

    ones = np.ones((points_xy.shape[0], 1))
    pts_h = np.hstack([points_xy, ones])  # Shape: (N, 3)

    projected_h = (H @ pts_h.T).T  # Shape: (N, 3)
    projected_xy = projected_h[:, :2] / projected_h[:, 2, np.newaxis]

    # Flip (x, y) → (y, x)
    projected_yx = projected_xy[:, ::-1].copy()
    return projected_yx


def project_bboxes_with_homography(bboxes, H):
    """
    Project bounding boxes from one image frame to another using a homography.
    Batched: all bbox corners are projected in a single matrix multiply.

    Parameters:
        bboxes: list/array of (min_y, min_x, max_y, max_x) tuples, length N
        H: 3x3 homography matrix

    Returns:
        list of (min_y, min_x, max_y, max_x) tuples in projected space
    """
    if len(bboxes) == 0:
        return []

    bboxes_arr = np.asarray(bboxes, dtype=np.float64)  # (N, 4)

    # Stack all corners into (2N, 2) array: [min_y, min_x] and [max_y, max_x] per bbox
    all_corners = np.empty((2 * len(bboxes_arr), 2), dtype=np.float64)
    all_corners[0::2] = bboxes_arr[:, :2]  # (min_y, min_x)
    all_corners[1::2] = bboxes_arr[:, 2:]  # (max_y, max_x)

    # Single batched projection
    projected = project_points_with_homography(all_corners, H)  # (2N, 2) in (y, x)

    # Reshape to (N, 2, 2) — each bbox has 2 projected corners
    projected = projected.reshape(-1, 2, 2)

    # Extract min/max per bbox
    projected_bboxes = []
    for i in range(len(projected)):
        y1 = int(np.floor(projected[i, :, 0].min()))
        x1 = int(np.floor(projected[i, :, 1].min()))
        y2 = int(np.ceil(projected[i, :, 0].max()))
        x2 = int(np.ceil(projected[i, :, 1].max()))
        projected_bboxes.append((y1, x1, y2, x2))

    return projected_bboxes


def suppress_bboxes_on_diff_map(diff_map, projected_bboxes, margin_px=0):
    """
    Zero out regions in the diff map around projected bounding boxes.

    Parameters:
        diff_map: 2D array (modified in-place)
        projected_bboxes: list of (min_y, min_x, max_y, max_x)
        margin_px: pixel margin to expand each bbox
    """
    h, w = diff_map.shape[:2]

    # Build a mask of all bbox regions (True = inside a bbox)
    bbox_mask = np.zeros((h, w), dtype=bool)
    for bbox in projected_bboxes:
        y1 = max(0, bbox[0] - margin_px)
        x1 = max(0, bbox[1] - margin_px)
        y2 = min(h, bbox[2] + margin_px)
        x2 = min(w, bbox[3] + margin_px)
        bbox_mask[y1:y2, x1:x2] = True

    # Compute mean excluding bbox regions
    outside = ~bbox_mask
    if outside.any():
        mean_value = diff_map[outside].mean()
    else:
        mean_value = diff_map.mean()  # fallback if bboxes cover entire image

    # Suppress
    diff_map[bbox_mask] = mean_value


def compute_diff_threshold_from_zscore(diff_map, z_score, min_threshold=0.0):
    """
    Compute an absolute pixel threshold from a z-score on the diff map.

    threshold = max(mean + z_score * std, min_threshold)

    Parameters:
        diff_map: 2D array (already suppressed)
        z_score: number of standard deviations above the mean
        min_threshold: minimum floor value (e.g. sensor measurement error)

    Returns:
        float: absolute threshold value
    """
    # --- Z-score approach ---
    mu = diff_map.mean()
    sigma = diff_map.std()
    threshold = mu + z_score * sigma

    # --- MAD-based approach (more robust to outlier fire pixels) ---
    # median = np.median(diff_map)
    # mad = np.median(np.abs(diff_map - median))
    # threshold = median + z_score * 1.4826 * mad  # 1.4826 scales MAD to match std for Gaussian

    return max(threshold, min_threshold)


def preprocess_images(image1, image2=None, applying=False):
    img1 = image1.astype(np.float32)
    if applying:
        img1 = img1 - img1.mean()
    if image2 is None:
        return img1
    img2 = image2.astype(np.float32)
    if applying:
        img2 = img2 - img2.mean()
    return img1, img2


def compute_positive_difference(img1, img2):
    diff = img2 - img1
    # diff = np.maximum(diff, 0)
    return diff


def postprocess_difference_map(diff, img2, threshold=None, temp_threshold=None):
    """
    Post-processes the difference map by zeroing out pixels that do not meet the conditions.

    Parameters:
        diff: difference map (2D array)
        img2: reference image (same shape)
        threshold: minimum value in diff to keep (None = no filtering)
        temp_threshold: intensity threshold based on mean of img2 (None = no filtering)

    Returns:
        diff: diff map after zeroing out irrelevant pixels
    """
    if threshold is not None:
        diff[diff <= threshold] = 0

    if temp_threshold is not None:
        temp_mask = img2 <= (np.median(img2) + temp_threshold)
        diff[temp_mask] = 0

    return diff


def find_cluster_centers_conditional(
    diff_map,
    threshold=10,
    half_kernel_size_x=1,
    half_kernel_size_y=1,
    min_samples=2,
    min_contrast=10
):

    mask = (diff_map > threshold).astype(np.uint8)
    if mask.sum() == 0:
        return [], [], []

    # Morphological closing with rectangular kernel (accounts for non-uniform GSD)
    hk_x = int(round(half_kernel_size_x))
    hk_y = int(round(half_kernel_size_y))
    if hk_x >= 1 or hk_y >= 1:
        hk_x = max(hk_x, 1)
        hk_y = max(hk_y, 1)
        kernel_w = 2 * hk_x + 1  # width (x direction, columns)
        kernel_h = 2 * hk_y + 1  # height (y direction, rows)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    else:
        grouped = mask

    # Connected components
    num_labels, labels_raw = cv2.connectedComponents(grouped, connectivity=8)

    # print(f"num_labels: {num_labels}")

    # Active pixels only (same logic as before)
    active = mask.astype(bool)

    # === Collect all relevant points ONCE ===
    points = np.argwhere((labels_raw > 0) & active)
    if len(points) == 0:
        return [], [], []

    point_labels = labels_raw[points[:, 0], points[:, 1]]
    values = diff_map[points[:, 0], points[:, 1]]

    # === Sort by label for grouping ===
    order = np.argsort(point_labels)
    points = points[order]
    point_labels = point_labels[order]
    values = values[order]

    # === Split into clusters ===
    split_idx = np.where(np.diff(point_labels) != 0)[0] + 1
    points_split = np.split(points, split_idx)
    values_split = np.split(values, split_idx)
    labels_split = np.split(point_labels, split_idx)

    # Output containers
    label_map = np.full(diff_map.shape, -1, dtype=int)
    centers = []
    bboxes = []
    next_label = 0

    # === Process each cluster ===
    for cluster_points, cluster_values, cluster_labels in zip(points_split, values_split, labels_split):

        if len(cluster_points) < min_samples:
            continue

        # Assign contiguous label
        label_map[cluster_points[:, 0], cluster_points[:, 1]] = next_label

        # --- center calculation ---
        z = (cluster_values.max() - cluster_values.mean()) / (cluster_values.std() + 1e-6)

        if z >= min_contrast:
            center = cluster_points[np.argmax(cluster_values)]
        else:
            weights = cluster_values - cluster_values.min() + 1e-6
            weights /= weights.sum()
            center = np.average(cluster_points, axis=0, weights=weights)

        centers.append(tuple(center))

        # --- bbox ---
        min_y, min_x = cluster_points.min(axis=0)
        max_y, max_x = cluster_points.max(axis=0)
        bboxes.append((min_y, min_x, max_y + 1, max_x + 1))

        next_label += 1

    if len(centers) == 0:
        return [], [], []

    # print(f"len(centers): {len(centers)}")

    return centers, label_map, bboxes


def compute_cluster_size_maxval(label_map, image, pixel_area_m2):
    cluster_info = {}

    flat_labels = label_map.ravel()
    flat_image = image.ravel()

    # remove background (-1)
    valid = flat_labels >= 0
    flat_labels = flat_labels[valid]
    flat_image = flat_image[valid]

    # cluster sizes (pixel count)
    counts = np.bincount(flat_labels)

    # max per label
    max_vals = np.zeros_like(counts, dtype=float)
    mean_vals = np.zeros_like(counts, dtype=float)
    for label in range(len(counts)):
        if counts[label] > 0:
            max_vals[label] = flat_image[flat_labels == label].max()
            mean_vals[label] = flat_image[flat_labels == label].mean()
        else:
            max_vals[label] = 0
            mean_vals[label] = 0

    for label in range(len(counts)):
        size_m2 = int(counts[label]) * pixel_area_m2
        cluster_info[label] = {
            "size": size_m2,
            "max_val": float(max_vals[label]),
            "mean_val": float(mean_vals[label]),
        }

    return cluster_info


def compute_cluster_scores(
    cluster_info_filtered,
    norm_size=5**2,
    norm_intensity=200,
    weights=(0.5, 0.5, 10, 0.4),
):
    """
    Compute confidence score [0,1] for each detected fire cluster
    using precomputed cluster_info.

    Parameters
    ----------
    cluster_info_filtered : dict
        {label: {"size": m2, "mean_val": float, "max_val": float}}

    norm_size : float
        Area normalization constant [m²].

    norm_intensity : float
        Intensity normalization constant.

    weights : tuple (wI, wA, k, t)
        wI : weight of intensity
        wA : weight of area
        k  : sigmoid steepness
        t  : sigmoid threshold

    Returns
    -------
    dict
        {label: score ∈ [0,1]}
    """

    scores = {}
    wI, wA, k, t = weights

    for label, info in cluster_info_filtered.items():

        fire_area_m2 = info["size"]
        mean_intensity = info["mean_val"]

        # --- Area term ---
        A_norm = np.clip(fire_area_m2 / norm_size, 0, 1.5)

        # --- Intensity term ---
        I_norm = np.clip(mean_intensity / norm_intensity, 0, 1.5)

        # --- Fusion ---
        grade = wI * I_norm + wA * A_norm

        # --- Sigmoid shaping ---
        score = 1.0 / (1.0 + np.exp(-k * (grade - t)))
        score = np.clip(score, 0, 1)

        scores[label] = round(float(score), 3)

    return scores


def generate_uniform_grid(h, w, points_num):
    # Compute number of points along y and x axis maintaining the aspect ratio
    ratio = h / w
    n_y = int(np.round(np.sqrt(points_num * ratio)))
    n_x = int(np.round(points_num / n_y))
    
    # Safety correction to ensure total points = points_num
    while n_x * n_y > points_num:
        if n_x > n_y:
            n_x -= 1
        else:
            n_y -= 1
    while n_x * n_y < points_num:
        if n_x < n_y:
            n_x += 1
        else:
            n_y += 1

    # Generate grid coordinates
    ys = np.linspace(0, h - 1, n_y, dtype=int)
    xs = np.linspace(0, w - 1, n_x, dtype=int)
    points = np.array([(y, x) for y in ys for x in xs])

    return points


def compute_worst_case_gsd(h, theta_deg, hfov_deg, n_y, n_x):
    """
    Compute worst-case (maximum) GSD in x and y directions under oblique viewing.

    Based on a pinhole camera model with locally flat ground plane.
    The worst-case vertical GSD occurs at the far edge of the image (phi = VFOV/2),
    and the worst-case horizontal GSD at the far row and horizontal edge (psi = HFOV/2).

    Formulation:
        GSD_y_max ≈ h * [tan(θ + VFOV/2) - tan(θ + VFOV/2 - VFOV/N_y)]
        GSD_x_max ≈ h / cos(θ + VFOV/2) * [tan(HFOV/2) - tan(HFOV/2 - HFOV/N_x)]

    Parameters
    ----------
    h : float
        Flight altitude [meters].
    theta_deg : float
        Payload viewing angle from nadir [degrees]. 0 = looking straight down.
    hfov_deg : float
        Horizontal field of view [degrees].
    n_y : int
        Image height in pixels.
    n_x : int
        Image width in pixels.

    Returns
    -------
    gsd_x_max : float
        Worst-case horizontal GSD [meters/pixel].
    gsd_y_max : float
        Worst-case vertical GSD [meters/pixel].
    """
    theta = np.radians(theta_deg)
    hfov = np.radians(hfov_deg)

    # Compute VFOV from HFOV using sensor aspect ratio
    aspect_ratio = n_y / n_x
    vfov = 2 * np.arctan(np.tan(hfov / 2) * aspect_ratio)

    # Far-edge vertical angle (clip to avoid degenerate past-horizontal cases)
    far_edge_angle = theta + vfov / 2
    max_valid_angle = np.radians(89.0)
    far_edge_angle = min(far_edge_angle, max_valid_angle)

    # Worst-case vertical GSD at far edge (phi = VFOV/2)
    gsd_y_max = h * (np.tan(far_edge_angle) - np.tan(far_edge_angle - vfov / n_y))

    # Worst-case horizontal GSD at far row and horizontal edge (psi = HFOV/2)
    psi_max = hfov / 2
    gsd_x_max = (h / np.cos(far_edge_angle)) * (np.tan(psi_max) - np.tan(psi_max - hfov / n_x))

    return gsd_x_max, gsd_y_max


def camera_angle_from_vertical(
        platform_roll_deg: float,
        platform_pitch_deg: float,
        platform_yaw_deg: float,
        sensor_azimuth_deg: float,
        sensor_elevation_deg: float,
) -> float:
    """
    Compute the angle between the camera main optical axis and the vertical
    direction toward the ground (Down / Nadir).

    Coordinate system:
    - World frame: NED (North-East-Down), right-handed.
      X = North (Forward), Y = East (Right), Z = Down.

    - Camera canonical viewing direction (at 0,0,0):
      Points along the X-axis (Forward).

    Parameters
    ----------
    platform_roll_deg : float
        Platform roll (Tag 7). Positive = Right wing down.
    platform_pitch_deg : float
        Platform pitch (Tag 6). Positive = Nose up.
    platform_yaw_deg : float
        Platform heading (Tag 5). 0=North, 90=East.
    sensor_azimuth_deg : float
        Sensor relative yaw (Tag 18). 0=Forward, Positive=Clockwise.
    sensor_elevation_deg : float
        Sensor relative elevation (Tag 19). Negative=Down.

    Returns
    -------
    float
        Angle in degrees between the camera viewing axis and the vertical
        direction toward the ground (Down / +Z).
        0.0 = Camera looking straight down.
    """

    # --- 1. Platform Rotation ---
    # "Start with Yaw, then Pitch, then Roll."
    # Axes mapping: Yaw=Z, Pitch=Y, Roll=X.
    # Scipy convention 'zyx' corresponds to Intrinsic Z -> Y -> X.
    platform2world = R.from_euler(
        'ZYX', # TODO: Make sure intrinsic rotations or extrinsic rotations
        [platform_yaw_deg, platform_pitch_deg, platform_roll_deg],
        degrees=True
    ).as_matrix()

    # --- 2. Sensor Rotation ---
    # Sensor rotates relative to the platform body.
    # Order: Azimuth (Yaw around Z) -> Elevation (Pitch around Y).
    # Negative elevation is down.
    # (Scipy 'y' rotation of -90 on vector [1,0,0] results in [0,0,1] which is Down/+Z,
    # so the sign convention is handled correctly naturally).
    sensor2platform = R.from_euler(
        'ZY', # TODO: Make sure intrinsic rotations or extrinsic rotations
        [sensor_azimuth_deg, sensor_elevation_deg],
        degrees=True
    ).as_matrix()

    # --- 3. Canonical Camera Vector ---
    # "0 degrees forward along the longitudinal axis".
    # In NED, Longitudinal/Forward is +X.
    v_camera_canonical = np.array([1.0, 0.0, 0.0])

    # --- 4. Calculate World Vector ---
    # Apply sensor rotation, then platform rotation
    v_sensor_body = sensor2platform @ v_camera_canonical
    v_sensor_world = platform2world @ v_sensor_body

    # --- 5. Define Down Vector ---
    # In NED, Down is +Z
    v_down = np.array([0.0, 0.0, 1.0])

    # --- 6. Calculate Angle ---
    # Dot product: a . b = |a||b| cos(theta)
    # Since vectors are normalized (length 1), cos(theta) = a . b
    dot_prod = np.dot(v_sensor_world, v_down)

    # Clip for numerical stability
    cos_theta = np.clip(dot_prod, -1.0, 1.0)

    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    return angle_deg
