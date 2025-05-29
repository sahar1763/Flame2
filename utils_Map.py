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

# # === 2. Convert grid index (i, j) to image coordinates (u, v) ===
# def grid_to_image(i, j, H_grid_to_world, H_world_to_image):
#     """
#     Converts a grid cell (i, j) to its corresponding image pixel (u, v),
#     using two homographies: grid→world, world→image.
#     """
#     # Convert grid index to homogeneous point
#     point_grid = np.array([j, i, 1.0], dtype=np.uint8)

#     # Map to world coordinates
#     point_world = H_grid_to_world @ point_grid
#     point_world /= point_world[2]
#     lon, lat = point_world[0], point_world[1]

#     # Map to image coordinates
#     point_world_hom = np.array([lon, lat, 1.0], dtype=np.uint8)
#     point_image = H_world_to_image @ point_world_hom
#     point_image /= point_image[2]
#     u, v = point_image[0], point_image[1]

#     return u, v

# === 3. Map entire grid to image using homographies and remap ===
def warp_image_to_grid(grid, image, H_grid_to_world, H_world_to_image, grid_height, grid_width):
    """
    Projects the image onto the grid and returns a new updated grid (copy),
    where only valid points (inside image bounds) are updated.

    This version is optimized for memory by using float32 and np.indices.
    """
    # Copy the original grid to update only valid regions
    new_grid = grid.copy()

    # 1. Generate grid coordinates (i, j)
    ii, jj = np.indices((grid_height, grid_width), dtype=np.float32)  # shape: (2, H, W)

    # 2. Flatten and convert to homogeneous grid coordinates (j, i, 1)
    ones = np.ones_like(ii)
    grid_points = np.stack([jj, ii, ones], axis=0).reshape(3, -1)  # shape: (3, H*W)

    # 3. Grid → World
    world_coords = H_grid_to_world @ grid_points
    world_coords /= world_coords[2, :]  # normalize homogeneous

    # 4. World → Image
    image_coords = H_world_to_image @ world_coords
    image_coords /= image_coords[2, :]  # normalize homogeneous

    # 5. Reshape u, v
    u = image_coords[0, :].reshape((grid_height, grid_width)).astype(np.float32)
    v = image_coords[1, :].reshape((grid_height, grid_width)).astype(np.float32)

    # 6. Sample from image using OpenCV remap
    sampled_values = cv2.remap(image.astype(np.float32), u, v,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)

    # 7. Validity mask: only update values that fall inside the image
    valid_mask = (u >= 0) & (u < image.shape[1]) & (v >= 0) & (v < image.shape[0])
    new_grid[valid_mask] = sampled_values[valid_mask]

    return new_grid


# === 5. Find subgrid coordinates ===
def get_subgrid_bounds_precise(pts_world, lon_origin, lat_origin, resolution, grid_width=None, grid_height=None):
    """
    Given 4 world points (lon, lat), return the min/max grid *locations* (not indices),
    by projecting to grid coordinates and flooring/ceiling the outer bounds.

    Returns:
        j_min, j_max, i_min, i_max — grid locations (not necessarily integer indices)
    """
    lon = np.asarray(pts_world[:, 0])
    lat = np.asarray(pts_world[:, 1])

    # Map to real-valued grid locations
    j_float = (lon - lon_origin) / resolution
    i_float = (lat - lat_origin) / resolution

    padding_marg = 10
    # Apply floor/ceil to get enclosing bounds
    j_min = np.floor(np.min(j_float)) - padding_marg
    j_max = np.ceil(np.max(j_float)) + padding_marg
    i_min = np.floor(np.min(i_float)) - padding_marg
    i_max = np.ceil(np.max(i_float)) + padding_marg

    # Clip to grid size if specified
    if grid_width is not None:
        j_min = max(j_min, 0)
        j_max = min(j_max, grid_width - 1)

    if grid_height is not None:
        i_min = max(i_min, 0)
        i_max = min(i_max, grid_height - 1)

    return j_min, j_max, i_min, i_max

# # === 5. Creating subgrid ===
# def extract_subgrid_from_bounds(grid, j_min, j_max, i_min, i_max):
#     """
#     Cuts a subgrid from the full grid using float-based grid bounds.

#     Parameters:
#         grid: 2D array [height, width]
#         j_min, j_max, i_min, i_max: float grid locations

#     Returns:
#         subgrid: 2D array (sliced from grid)
#         i_start, i_end, j_start, j_end: integer indices used
#     """
#     # Convert float positions to integer indices
#     j_start = int(j_min)
#     j_end   = int(j_max)
#     i_start = int(i_min)
#     i_end   = int(i_max)

#     # Cut the subgrid
#     subgrid = grid[i_start:i_end, j_start:j_end]

#     return subgrid, i_start, i_end, j_start, j_end


# def create_subgrid_from_bounds(lon_min, lon_max, lat_min, lat_max, resolution):
#     """
#     Cuts a subgrid from the full grid using float-based grid bounds.

#     Parameters:
#         grid: 2D array [height, width]
#         j_min, j_max, i_min, i_max: float grid locations

#     Returns:
#         subgrid: 2D array (sliced from grid)
#         i_start, i_end, j_start, j_end: integer indices used
#     """

#     # Determine shape of the subgrid
#     subgrid_width = int((lon_max - lon_min) / resolution)
#     subgrid_height = int((lat_max - lat_min) / resolution)

#     # Create subgrid filled with NaNs
#     subgrid = np.full((subgrid_height, subgrid_width), np.nan) #, dtype=np.uint8)

#     return subgrid
    

## Find Diff

def preprocess_images(image1, image2, applying=0):
    img1 = image1.astype(np.uint8)
    img2 = image2.astype(np.uint8)

    if applying > 0:
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()

    return img1, img2

def compute_positive_difference(img1, img2):
    diff = img2 - img1
    diff[np.isnan(diff)] = 0
    diff[diff < 0] = 0
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



def extract_and_resample_subgrid_interp(grid, resolution, res, j_min, j_max, i_min, i_max):
    """
    Extracts a region from a high-resolution grid and resamples it using interpolation to a new resolution.

    Parameters:
        grid: 2D numpy array (original high-res grid)
        resolution: original grid resolution (e.g., 1 meter per cell)
        res: desired new resolution (e.g., 5 meters per cell)
        j_min, j_max, i_min, i_max: float bounds (in grid coordinates)

    Returns:
        subgrid_interp: 2D numpy array resampled to new resolution
        lon_new, lat_new: 1D arrays with the coordinates of the new grid
    """

    # Step 1: Convert float bounds to integer indices (safe)
    j_start = max(0, int(np.floor(j_min)))
    j_end   = min(grid.shape[1], int(np.ceil(j_max)))
    i_start = max(0, int(np.floor(i_min)))
    i_end   = min(grid.shape[0], int(np.ceil(i_max)))

    # Step 2: Extract subgrid
    subgrid_highres = grid[i_start:i_end, j_start:j_end]

    # Step 3: Compute lat/lon coordinate axes in world units
    lat_vals = (np.arange(i_start, i_end) * resolution).astype(np.float64)
    lon_vals = (np.arange(j_start, j_end) * resolution).astype(np.float64)

    # Step 4: Build interpolator over the high-res subgrid
    interpolator = RegularGridInterpolator(
        (lat_vals, lon_vals),
        subgrid_highres,
        bounds_error=False,
        fill_value=255
    )

    # Step 5: Carefully generate coarse-resolution grid inside valid bounds
    def safe_arange(start, stop, step):
        """Like np.arange, but ensures last value doesn't exceed stop."""
        vals = []
        v = start
        while v <= stop:
            vals.append(v)
            v += step
        return np.array(vals, dtype=np.float64)

    lat_new = safe_arange(lat_vals[0], lat_vals[-1], res)
    lon_new = safe_arange(lon_vals[0], lon_vals[-1], res)

    # Step 6: Create sampling meshgrid and interpolate
    lon_mesh, lat_mesh = np.meshgrid(lon_new, lat_new)
    sample_points = np.stack([lat_mesh.ravel(), lon_mesh.ravel()], axis=-1)

    # Step 7: Interpolate
    subgrid_interp = interpolator(sample_points).reshape(lat_mesh.shape)

    return subgrid_interp, lon_new, lat_new



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
    
    return num_spots_range



# def compute_vfov_from_hfov(hfov_deg, width, height):
#     hfov_rad = np.radians(hfov_deg)
#     aspect_ratio = height / width
#     vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * aspect_ratio)
#     return np.degrees(vfov_rad)

def get_ground_corners(x, y, h, theta_deg, phi_deg, hfov_deg, width=1280, height=720):
    vfov_deg = compute_vfov_from_hfov(hfov_deg, width, height)

    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    hfov = np.radians(hfov_deg)
    vfov = np.radians(vfov_deg)

    corners = np.array([
        [np.tan(hfov/2),  np.tan(vfov/2), -1],
        [-np.tan(hfov/2),  np.tan(vfov/2), -1],
        [-np.tan(hfov/2), -np.tan(vfov/2), -1],
        [np.tan(hfov/2), -np.tan(vfov/2), -1],
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

# def compute_polygon_area(points):
#     # Shoelace formula for area of polygon
#     x = points[:, 0]
#     y = points[:, 1]
#     return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# def plot_drone_view(corners, drone_pos, range_xy=15000):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.plot(*drone_pos, 'ro', label='Drone Position')

#     polygon = np.vstack([corners, corners[0]])
#     ax.plot(polygon[:, 0], polygon[:, 1], 'b-', label='Camera Footprint')
#     ax.fill(polygon[:, 0], polygon[:, 1], color='lightblue', alpha=0.4)

#     # קביעת טווח הצירים לפי המרכז שאתה רוצה
#     ax.set_xlim(- range_xy, range_xy)
#     ax.set_ylim(- range_xy, range_xy)

#     ax.set_aspect('equal')
#     ax.grid(True)
#     ax.legend()
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_title("Drone Camera Ground Footprint")
#     plt.show()

def find_cluster_centers_conditional(diff_map, threshold=10, eps=1.5, min_samples=2, min_contrast=10):
    """
    Applies DBSCAN on a diff map and returns the center of each cluster.
    If the contrast in the cluster (max - min) >= min_contrast, the center is the hottest point.
    Otherwise, the center is the geometric mean.

    Parameters:
        diff_map: 2D array
        threshold: only consider diff values above this
        eps: DBSCAN neighborhood radius
        min_samples: DBSCAN min points per cluster
        min_contrast: min difference (max - min) in a cluster to use the hottest point

    Returns:
        centers: list of (i, j) tuples
        label_map: same-shape array with cluster labels
    """
    active_pixels = np.argwhere(diff_map > threshold)
    if len(active_pixels) == 0:
        return [], np.full_like(diff_map, -1)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(active_pixels)
    labels_flat = clustering.labels_

    label_map = np.full(diff_map.shape, -1, dtype=int)
    for idx, (i, j) in enumerate(active_pixels):
        label_map[i, j] = labels_flat[idx]

    centers = []
    for label in np.unique(labels_flat):
        if label == -1:
            continue  # skip noise

        cluster_points = active_pixels[labels_flat == label]
        values = diff_map[cluster_points[:, 0], cluster_points[:, 1]]
        contrast = values.max() - 2*values.mean()

        if contrast >= 0:
            # Use hottest point
            hottest_idx = np.argmax(values)
            center = cluster_points[hottest_idx]
        else:
            # Use geometric center
            center = cluster_points.mean(axis=0)

        centers.append(tuple(center))

    return centers, label_map
