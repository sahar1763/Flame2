import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import DBSCAN


# === 1. Create homography matrix ===
def create_homography(pts_dst, pts_src):
    """
    Computes the homography matrix from pts_src to pts_dst.
    """
    H, _ = cv2.findHomography(pts_src, pts_dst)
    return H


def preprocess_images(image1, image2, applying=0):
    img1 = image1.astype(np.uint8)
    img2 = image2.astype(np.uint8)

    if applying > 0:
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
        temp_mask = img2 <= (img2.mean() + temp_threshold)
        diff[temp_mask] = 0

    return diff

def project_points_with_homography(corners, H):
    # Convert input 2D corner points to homogeneous coordinates by appending a column of ones.
    # corners: array of shape (N, 2), where N is the number of points.
    # corners_h: array of shape (N, 3)
    ones = np.ones((corners.shape[0], 1))
    corners_h = np.hstack([corners, ones])  # Shape: (N, 3)

    # Apply the homography transformation matrix H to the homogeneous coordinates.
    # H is a 3x3 matrix, and the result is a set of projected homogeneous coordinates.
    projected_h = (H @ corners_h.T).T  # Shape: (N, 3)

    # Convert back from homogeneous to 2D pixel coordinates by dividing x and y by the scale (z).
    projected_pixels = projected_h[:, :2] / projected_h[:, 2, np.newaxis]

    # Return the projected 2D pixel coordinates as a (N, 2) array.
    return projected_pixels



def compute_vfov_from_hfov(hfov_deg, width, height):
    """
    Computes the vertical field of view (vFOV) in degrees, given the horizontal field of view (hFOV)
    and the image's width and height.

    Parameters:
        hfov_deg (float): Horizontal field of view in degrees.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        vfov_deg (float): Vertical field of view in degrees.
    """
    # Convert horizontal FOV from degrees to radians
    hfov_rad = np.radians(hfov_deg)
    
    # Compute aspect ratio (height divided by width)    
    aspect_ratio = height / width

    # Use trigonometric identity to calculate vertical FOV in radians
    vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * aspect_ratio)

    # Convert vertical FOV back to degrees
    return np.degrees(vfov_rad)



def create_synthetic_image_with_clusters(image_height, image_width,
                                         background_range=(3, 7),
                                         cluster_value=200,
                                         num_clusters_range=(1, 4),
                                         cluster_radius_range=(1, 9)):
    """
    Creates a synthetic image with uniform random background and a few bright Gaussian clusters.

    Parameters:
        image_height, image_width: dimensions of the image
        background_range: range for background pixel values (uniform)
        cluster_value: peak intensity for Gaussian clusters
        num_clusters_range: range of number of clusters to add (inclusive)
        cluster_radius_range: range of radius (in pixels) for each cluster (inclusive)

    Returns:
        A 2D numpy array representing the synthetic image
    """
    # 1. Generate uniform background values
    image = np.random.uniform(*background_range, size=(image_height, image_width)).astype(np.uint8)

    # 2. Randomly choose how many clusters to insert
    num_clusters = np.random.randint(num_clusters_range[0], num_clusters_range[1] + 1)

    cluster_centers = []

    for _ in range(num_clusters):
        # 3. Random cluster center (cx, cy)
        cx = np.random.randint(0, image_width)
        cy = np.random.randint(0, image_height)

        # Save the center
        cluster_centers.append((cx, cy))

        # 4. Random radius for the Gaussian cluster
        radius = np.random.randint(cluster_radius_range[0], cluster_radius_range[1] + 1)

        # 5. Compute 2D Gaussian mask centered at (cx, cy)
        y, x = np.meshgrid(np.arange(image_height), np.arange(image_width), indexing='ij')
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        cluster_value = np.random.randint(cluster_value - 50, cluster_value + 50)
        gaussian_blob = cluster_value * np.exp(-dist_sq / (2 * (radius ** 2)))

        # 6. Update the image with the cluster (take maximum where overlapping)
        image = np.maximum(image, gaussian_blob)
        

    return image, num_clusters, cluster_centers



def add_uniform_spots(image,
                      value_range=(95, 105),
                      spot_radius_range=(10, 30),
                      num_spots_range=(1, 4)):
    """
    Adds circular patches to the image with uniform random noise around a target value.

    Parameters:
        image: 2D numpy array (modified in-place)
        value_range: range of values for uniform noise inside each spot
        spot_radius_range: min/max radius of each spot
        num_spots_range: number of spots to add
    """
    height, width = image.shape
    num_spots = np.random.randint(*num_spots_range)

    for _ in range(num_spots):
        # Random center
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)

        # Random radius
        radius = np.random.randint(*spot_radius_range)

        # Build mask of circular region
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        mask = dist_sq <= radius ** 2

        # Add uniform random values in the specified range to that region
        patch = np.random.uniform(*value_range, size=image.shape)
        image[mask] = patch[mask]
    
    return num_spots


def get_ground_corners(x, y, h, theta_deg, phi_deg, hfov_deg, width=1280, height=720):
    vfov_deg = compute_vfov_from_hfov(hfov_deg, width, height)

    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    hfov = np.radians(hfov_deg)
    vfov = np.radians(vfov_deg)

    corners = np.array([
        [ -np.tan(hfov / 2),np.tan(vfov / 2), -1],
        [ np.tan(hfov / 2),np.tan(vfov / 2), -1],
        [ -np.tan(hfov / 2),-np.tan(vfov / 2), -1],
        [ np.tan(hfov / 2),-np.tan(vfov / 2), -1],
    ])

    dirs = corners / np.linalg.norm(corners, axis=1, keepdims=True)

    R_yaw = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi),  np.cos(phi)]
    ])

    R = R_yaw @ R_pitch
    world_dirs = dirs @ R.T

    corners_world = []
    for d in world_dirs:
        scale = -h / d[2]
        point = np.array([x, y, h]) + scale * d
        corners_world.append(point[:2])
    return np.array(corners_world)


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
        return [], [], np.full_like(diff_map, -1)

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
        bboxes.append((min_i, min_j, max_i, max_j))

    return centers, label_map, bboxes


def compute_cluster_scores(label_map, image1, GSD, norm_size=5**2, norm_intensity=200):
    """
    Computes a score for each labeled cluster in the label_map based on pixel intensity.

    Parameters:
        label_map (np.ndarray): 2D array with cluster labels (e.g., from DBSCAN), shape (H, W)
        image1 (np.ndarray): Grayscale or single-channel image, shape (H, W)
        GSD (float): Ground Sample Distance (meters/pixel)

    Returns:
        dict: {label: score}, excluding label -1 (background/noise)
    """
    scores = {}
    unique_labels = np.unique(label_map)

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise label

        mask = (label_map == label)
        total_intensity = np.sum(image1[mask])
        denom = norm_size * norm_intensity / (GSD ** 2)
        score = total_intensity / denom
        score = np.clip(score, 0, 1)
        scores[label] = round(score, 1)


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