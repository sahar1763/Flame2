import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R


def create_homography(pts_dst, pts_src):
    """
    Computes the homography matrix from pts_src to pts_dst.
    """
    H, _ = cv2.findHomography(pts_src, pts_dst)
    return H


def preprocess_images(image1, image2, applying=False):
    img1 = image1.astype(np.uint8)
    img2 = image2.astype(np.uint8)

    if applying:
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()

    return img1, img2


def compute_positive_difference(img1, img2):
    diff = img2 - img1
    # diff[np.isnan(diff)] = 0
    diff[img2 < img1] = 0
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
        temp_mask = img2 <= (img2.median() + temp_threshold)
        diff[temp_mask] = 0

    return diff


def find_cluster_centers_conditional(diff_map, threshold=10, eps=1.5, min_samples=2, min_contrast=10):
    """
    Applies DBSCAN on a diff map and returns:
    - the center of each cluster, chosen conditionally (hottest point or geometric center),
    - the BBox of each cluster,
    - the full label map.

    Parameters:
        diff_map: 2D array
        threshold: only consider diff values above this
        eps: DBSCAN neighborhood radius
        min_samples: DBSCAN min points per cluster
        min_contrast: minimum contrast (custom condition) to prefer hottest point

    Returns:
        centers: list of (i, j) tuples (float)
        bboxes: list of (min_i, min_j, max_i, max_j)
        label_map: 2D array same shape as diff_map with cluster labels
    """
    active_pixels = np.argwhere(diff_map > threshold)
    if len(active_pixels) == 0:
        return [], np.full_like(diff_map, -1), []

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(active_pixels)
    labels_flat = clustering.labels_

    label_map = np.full(diff_map.shape, -1, dtype=int)
    for idx, (i, j) in enumerate(active_pixels):
        label_map[i, j] = labels_flat[idx]

    centers = []
    bboxes = []

    for label in np.unique(labels_flat):
        if label == -1:
            continue  # skip noise

        cluster_points = active_pixels[labels_flat == label]
        values = diff_map[cluster_points[:, 0], cluster_points[:, 1]]

        # Determine center
        contrast = values.max() - 2 * values.mean()
        if contrast >= min_contrast:
            hottest_idx = np.argmax(values)
            center = cluster_points[hottest_idx]
        else:
            center = cluster_points.mean(axis=0)

        centers.append(tuple(center))

        # Determine BBox
        min_i, min_j = cluster_points.min(axis=0)
        max_i, max_j = cluster_points.max(axis=0)
        bboxes.append((min_i, min_j, max_i+1, max_j+1))

    return centers, label_map, bboxes


def compute_cluster_scores(
    label_map,
    image1,
    GSD,
    norm_size=5**2,
    norm_intensity=200,
    weights=(0.5, 0.5, 10, 0.4),
):
    """
    Compute a confidence score ∈ [0,1] for each detected fire cluster based on
    its spatial size and thermal intensity.

    The score favors clusters that are both:
    - spatially large (big area on ground)
    - thermally strong (high mean pixel intensity)

    The process:
    1. Compute fire area in m² and normalize to [0,1]
    2. Compute mean intensity and normalize to [0,1]
    3. Fuse the two with a weighted sum
    4. Pass the result through a sigmoid to obtain a smooth confidence score

    Parameters
    ----------
    label_map : np.ndarray
        2D array with cluster labels (H×W).
        Label -1 is treated as background/noise.

    image1 : np.ndarray
        Single-channel thermal or intensity image (H×W).

    GSD : float
        Ground Sample Distance [meters/pixel].

    norm_size : float
        Area normalization constant [m²].
        Clusters larger than this saturate size contribution.

    norm_intensity : float
        Intensity normalization constant.
        Mean intensities higher than this saturate intensity contribution.

    weights : tuple (wI, wA, k, t)
        wI : weight of intensity contribution
        wA : weight of area contribution
        k  : sigmoid steepness (larger → sharper transition)
        t  : sigmoid threshold (center point)

    Returns
    -------
    dict
        {label: score} mapping, where score ∈ [0,1].
        Higher score means higher likelihood of a real fire.
    """

    scores = {}
    unique_labels = np.unique(label_map)
    wI, wA, k, t = weights

    for label in unique_labels:
        if label == -1:
            continue

        mask = (label_map == label)
        fire_area_pixel = np.sum(mask)

        if fire_area_pixel == 0:
            scores[label] = 0
            continue

        # --- Area term ---
        fire_area_m2 = fire_area_pixel * (GSD ** 2)
        # Clipping to avoid to dominant extreme values of one of the components in the sigmoid score function
        A_norm = np.clip(fire_area_m2 / norm_size, 0, 1.5)

        # --- Intensity term ---
        mean_intensity = image1[mask].mean() # TODO: Use max instead?
        # Clipping to avoid to dominant extreme values of one of the components in the sigmoid score function
        I_norm = np.clip(mean_intensity / norm_intensity, 0, 1.5)

        # --- Fusion ---
        grade = wI * I_norm + wA * A_norm

        # --- Sigmoid shaping ---
        score = 1.0 / (1.0 + np.exp(-k * (grade - t)))
        score = np.clip(score, 0, 1)

        scores[label] = round(score, 3)


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
