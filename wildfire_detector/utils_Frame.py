import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment


def project_points_with_homography(corners, H):
    # corners: array of shape (N, 2), where N is the number of points.
    ones = np.ones((corners.shape[0], 1))
    corners_h = np.hstack([corners, ones])  # Shape: (N, 3)

    projected_h = (H @ corners_h.T).T  # Shape: (N, 3)

    # Convert back from homogeneous to 2D pixel coordinates by dividing x and y by the scale (z).
    projected_pixels = projected_h[:, :2] / projected_h[:, 2, np.newaxis]

    # Return the projected 2D pixel coordinates as a (N, 2) array.
    return projected_pixels


def create_homography(pts_dst, pts_src):
    """
    Computes the homography matrix from pts_src to pts_dst.
    """
    H, _ = cv2.findHomography(pts_src, pts_dst)
    return H


def preprocess_images(image, applying=False):
    if applying:
        img = image.astype(np.float32)
        img = img - img.mean()
        img = np.clip(img, 0, 255)  # clamp negative values to 0 and max to 255
        img = img.astype(np.uint8)  # convert back to uint8
        return img
    return image


def compute_positive_difference(img1, img2):
    diff = img2 - img1
    diff = np.maximum(diff, 0)
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
        label_map: 2D array same shape as diff_map with cluster labels
        bboxes: list of (min_i, min_j, max_i, max_j)
    """
    active_pixels = np.argwhere(diff_map > threshold)
    if len(active_pixels) == 0:
        return [], [], []

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
        z = (values.max() - values.mean()) / (values.std() + 1e-6)
        if z >= min_contrast:
            hottest_idx = np.argmax(values)
            center = cluster_points[hottest_idx]
        else:
            # Compute weights: normalize to avoid extremely large differences
            weights = values - values.min() + 1e-6  # ensure positive
            weights /= weights.sum()  # normalize to sum=1

            # Compute weighted center
            center = np.average(cluster_points, axis=0, weights=weights)

        centers.append(tuple(center))

        # Determine BBox
        min_i, min_j = cluster_points.min(axis=0)
        max_i, max_j = cluster_points.max(axis=0)
        bboxes.append((min_i, min_j, max_i+1, max_j+1))

    return centers, label_map, bboxes


def associate_clusters(centers_phase1, centers_phase0, distance_threshold):
    """
    For each cluster center in phase1, find all candidate centers in phase0
    within distance_threshold.

    Parameters
    ----------
    centers_phase1 : list of (y, x)
    centers_phase0 : list of (y, x)
    distance_threshold : float (in pixels)

    Returns
    -------
    associations : dict
        {
            idx_phase1: [
                (idx_phase0, distance),
                ...
            ]
        }
        Each list is sorted from closest to farthest.
    """

    associations = {}

    if not centers_phase1:
        return {}

    if not centers_phase0:
        return {i: [] for i in range(len(centers_phase1))}

    p1 = np.array(centers_phase1, dtype=float)
    p0 = np.array(centers_phase0, dtype=float)

    for i, c1 in enumerate(p1):

        # Compute distances to all phase0 centers
        dists = np.linalg.norm(p0 - c1, axis=1)

        # Filter by threshold
        valid = np.where(dists <= distance_threshold)[0]

        # Build sorted list of (index, distance)
        matches = [(int(j), float(dists[j])) for j in valid]
        matches.sort(key=lambda x: x[1])

        associations[i] = matches

    return associations


def extract_orb_descriptors(image, cluster_centers, patch_size_px=31.0):
    """
    ORB descriptor extraction for real-time cluster tracking.

    Parameters:
    -----------
    patch_size_px : float
        ORB default is 31. This controls the 'size' of the area described.
    """

    patch_size_px = np.clip(patch_size_px, 31, 64)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    num_inputs = len(cluster_centers)
    if num_inputs == 0:
        return np.array([]), np.array([], dtype=bool)

    # Create KeyPoints
    input_kps = [
        cv2.KeyPoint(x=float(cx), y=float(cy), size=patch_size_px)
        for (cy, cx) in cluster_centers
    ]

    # ORB
    orb = cv2.ORB_create(patchSize=int(patch_size_px))
    keypoints_out, descriptors_out = orb.compute(gray, input_kps)

    final_descriptors = np.zeros((num_inputs, 32), dtype=np.uint8)
    valid_mask = np.zeros(num_inputs, dtype=bool)

    if descriptors_out is not None:

        out_coords = np.array([kp.pt for kp in keypoints_out])  # (M, 2)
        in_coords = np.array([(cx, cy) for (cy, cx) in cluster_centers])  # (N, 2)

        for i, pt in enumerate(out_coords):
            dist_sq = np.sum((in_coords - pt) ** 2, axis=1)
            original_idx = np.argmin(dist_sq)

            if dist_sq[original_idx] < 0.01:
                final_descriptors[original_idx] = descriptors_out[i]
                valid_mask[original_idx] = True

    return final_descriptors, valid_mask


def compute_cluster_cost_matrix(
    clusters_phase1,
    clusters_phase0,
    gsd,
    config,
):
    """
    Compute a cost matrix between clusters in two images.

    Parameters
    ----------
    clusters_phase1 : dict
        { idx: { "center": (y,x),
                  "descriptor": np.array(128,) or None,
                  "area": int,
                  "max_val": float } }

    clusters_phase0 : dict
        same structure as clusters_phase1

    gsd : float
        meters per pixel

    distance_threshold : float
        maximum allowed physical distance (meters).
        If exceeded, the score is set to infinity.

    w_dist, w_desc, w_area, w_maxval : float
        weights for each component. Should sum to 1.

    Returns
    -------
    cost_matrix : np.ndarray
        shape (len(clusters_phase1), len(clusters_phase0))
    """

    distance_threshold = config['cost_matrix']['max_distance']
    desc_score_threshold = config['cost_matrix']['desc_score_threshold']
    temp_threshold = config['cost_matrix']['temp_threshold']
    area_ratio_threshold = config['cost_matrix']['area_ratio_threshold']
    w_dist = config['cost_matrix']['scaling_weights']['dist']
    w_desc = config['cost_matrix']['scaling_weights']['desc']
    w_area = config['cost_matrix']['scaling_weights']['area']
    w_maxval = config['cost_matrix']['scaling_weights']['maxval']

    TOTAL_BITS = 256

    n1 = len(clusters_phase1)
    n0 = len(clusters_phase0)

    cost_matrix = np.zeros((n1, n0), dtype=float)

    for i, c1 in clusters_phase1.items():
        center1 = np.array(c1["center"])
        desc1 = c1["descriptor"]
        area1 = c1["area"]
        maxval1 = c1["max_val"]

        for j, c0 in clusters_phase0.items():
            center0 = np.array(c0["center"])
            desc0 = c0["descriptor"]
            area0 = c0["area"]
            maxval0 = c0["max_val"]

            # Physical distance
            dist_px = np.linalg.norm(center1 - center0)
            dist_m = dist_px * gsd

            # print(f"GSD: {gsd}")
            # print(f"dist_px: {dist_px}")
            # print(f"dist_m: {dist_m}")

            if dist_m > distance_threshold:
                cost_matrix[i, j] = 100
                continue

            dist_score = dist_m / distance_threshold

            # Descriptor similarity (cosine distance)
            if desc1 is None or desc0 is None or np.any(np.isnan(desc1)) or np.any(np.isnan(desc0)) \
                    or np.linalg.norm(desc1) == 0 or np.linalg.norm(desc0) == 0:
                desc_score = 1.0  # maximum distance if descriptor missing or invalid
            else:
                diff_bits = cv2.norm(desc1, desc0, cv2.NORM_HAMMING)
                desc_score = diff_bits / TOTAL_BITS

                if desc_score > desc_score_threshold:
                    cost_matrix[i, j] = 100

            # Area similarity
            area_ratio = min(area1, area0) / max(area1, area0)
            area_score = 1 - area_ratio
            if area_ratio > area_ratio_threshold:
                cost_matrix[i, j] = 100

            # Max value difference
            maxval_diff = abs(maxval1 - maxval0)
            maxval_score = maxval_diff / 255.0  # normalize to [0,1]
            if maxval_diff > temp_threshold:
                cost_matrix[i, j] = 100

            # Weighted total score
            total_score = (
                w_dist * dist_score +
                w_desc * desc_score +
                w_area * area_score +
                w_maxval * maxval_score
            )

            cost_matrix[i, j] = total_score

    return cost_matrix


def compute_cluster_size_maxval(label_map, image, gsd):
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
        size_m2 = int(counts[label]) * gsd**2
        cluster_info[label] = {
            "size": size_m2,
            "max_val": float(max_vals[label]),
            "mean_val": float(mean_vals[label]),
        }

    return cluster_info


def match_clusters_hungarian(cost_matrix):
    """
    Use the Hungarian algorithm to match clusters optimally based on the cost matrix.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Shape (num_clusters_phase1, num_clusters_phase0)

    Returns
    -------
    unmatched_mask : np.ndarray, bool
        True for clusters in phase1 that have NO valid match in phase0.
    matches : list of tuples
        List of (idx_phase1, idx_phase0) for matched clusters
    """
    num_phase1 = cost_matrix.shape[0]
    unmatched_mask = np.ones(num_phase1, dtype=bool)  # assume unmatched initially
    matches = []

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 50:
            unmatched_mask[r] = False
            matches.append((r, c))
        else:
            # cost is infinite → no valid match
            unmatched_mask[r] = True

    return unmatched_mask, matches


def filter_unmatched_clusters(unmatched_mask, centers, bboxes, cluster_info):
    """
    Filter out clusters that have no valid match based on unmatched_mask.

    Parameters
    ----------
    unmatched_mask : np.ndarray of bool
        True for clusters that have NO valid match.
    centers : list of tuples or np.ndarray
        Cluster centers (y,x) for phase1.
    bboxes : list of tuples
        Bounding boxes for clusters in phase1.
    label_map : np.ndarray, optional
        Label map corresponding to phase1 clusters. If provided, filtered as well.

    Returns
    -------
    filtered_centers : list
        Cluster centers with only matched clusters.
    filtered_bboxes : list
        Bounding boxes of matched clusters.
    filtered_label_map : np.ndarray or None
        Filtered label map (if provided), else None.
    """
    # Indices of matched clusters
    unmatched_indices = np.where(unmatched_mask)[0]

    filtered_centers = [centers[i] for i in unmatched_indices]
    filtered_bboxes = [bboxes[i] for i in unmatched_indices]
    filtered_cluster_info = {
        new_idx: cluster_info[old_idx]
        for new_idx, old_idx in enumerate(unmatched_indices)
    }

    return filtered_centers, filtered_bboxes, filtered_cluster_info


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
