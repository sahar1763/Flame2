import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from math import sqrt

from utils_Map import *

def main(grid_info, theta=150, phi=79.5, original_grid=None, i=0):
    """
    Runs the full image simulation and analysis pipeline for a given viewing angle.

    Parameters:
    - theta (float): Azimuth angle of the drone/camera view, in degrees.
    - phi (float): Tilt angle (elevation) of the camera, in degrees.
    - original_grid (2D array): High-resolution input grid representing the environment.
    - i (int): Identifier index used for saving plots or tracking runs.

    Returns:
    - result (dict): A dictionary containing statistics and metadata, such as:
        - Actual_num: Number of clusters inserted into the image.
        - Detected_num: Number of clusters detected by the algorithm.
        - Ratio: Detected/actual ratio.
        - clusters_center: Transformed cluster centers in grid coordinates.
    """
    # ================= Initial Parameters ==============
    # Finding corners of the image
    x, y = 0, 7500 # drone location
    h = 2500 # drone hight
    theta = theta  # yaw
    phi = phi    # pitch
    hfov = 17.5   # horizontal field of view
    
    lon_origin = grid_info['lon_origin']
    lon_max = grid_info['lon_max']
    lat_origin = grid_info['lat_origin']
    lat_max = grid_info['lat_max']
    resolution = grid_info['resolution']

    # Calculate the image corners coordinates
    corners = get_ground_corners(x, y, h, theta, phi, hfov)
    
    
    grid_width = int((lon_max - lon_origin) / resolution)
    grid_height = int((lat_max - lat_origin) / resolution)
    
    # Grid → world
    pts_grid = np.array([
        [0, 0],
        [grid_width - 1, 0],
        [grid_width - 1, grid_height - 1],
        [0, grid_height - 1]
    ], dtype=np.float32)
    
    pts_world_grid = np.array([
        [lon_origin, lat_origin],
        [lon_origin + (grid_width - 1) * resolution, lat_origin],
        [lon_origin + (grid_width - 1) * resolution, lat_origin + (grid_height - 1) * resolution],
        [lon_origin, lat_origin + (grid_height - 1) * resolution]
    ], dtype=np.float32)
    
    # World → image
    image_width = 1280
    image_height = 720
    
    pts_image = np.array([
          [0, 0],  # top-left
          [image_width - 1, 0],  # top-right
          [image_width - 1, image_height - 1],  # bottom-right
          [0, image_height - 1]  # bottom-left
      ], dtype=np.float32)
    
    pts_world_image = corners

    # ===============================================
    
    # Find homography  
    H_grid_to_world = create_homography(pts_world_grid, pts_grid)
    H_world_to_image = create_homography(pts_image, pts_world_image)

    # Creat image (including clusters)
    image, num_clusters, cluster_centers = create_synthetic_image_with_clusters(
        image_height=720,
        image_width=1280,
        background_range=(0, 0),
        cluster_value=200,
        num_clusters_range=(0, 4),
        cluster_radius_range=(1, 2)
    )
    
    # # Add uniform noisy spots
    # num_spots_range = add_uniform_spots(
    #     image,
    #     value_range=(50, 100),
    #     spot_radius_range=(3, 8),
    #     num_spots_range=(1, 4)
    # )
    num_spots = 0

    # ================= Grids comparison ==============
    # Cropping a Grid Patch Based the Image Bounding Box
    j_min, j_max, i_min, i_max = get_subgrid_bounds_precise(
        pts_world_image,
        lon_origin=lon_origin,
        lat_origin=lat_origin,
        resolution=resolution,
        grid_width=grid_width,
        grid_height=grid_height
    )

    if j_min >= j_max or i_min >= i_max:
        print(f"Skipping (theta={theta}, phi={phi}) due to empty subgrid.")
        return None
    
    lon_min = lon_origin + j_min * resolution
    lon_max = lon_origin + j_max * resolution
    lat_min = lat_origin + i_min * resolution
    lat_max = lat_origin + i_max * resolution

    # Finding the new sungrid resolution, base on the image
    max_dist = 0
    for a, b in combinations(corners, 2):
        dist = np.linalg.norm(a - b)
        if dist > max_dist:
            max_dist = dist
    
    res = max_dist / (1280 * 2 * sqrt(2))

    # Extract the new subgrid from the main grid
    subgrid, lon_vals, lat_vals = extract_and_resample_subgrid_interp(
        grid=original_grid,
        resolution=resolution,
        res=res,
        j_min=j_min, j_max=j_max,
        i_min=i_min, i_max=i_max
    )
    
    extent = [lon_min, lon_max, lat_min, lat_max]

    # Find Subgrid → world homography
    subgrid_width = subgrid.shape[1]
    subgrid_height = subgrid.shape[0]
    
    pts_subgrid = np.array([
        [0, 0],
        [subgrid_width - 1, 0],
        [subgrid_width - 1, subgrid_height - 1],
        [0, subgrid_height - 1]
    ], dtype=np.float32)
    
    pts_world_subgrid = np.array([
        [lon_min, lat_min],
        [lon_min + (subgrid_width - 1) * res, lat_min],
        [lon_min + (subgrid_width - 1) * res, lat_min + (subgrid_height - 1) * res],
        [lon_min, lat_min + (subgrid_height - 1) * res]
    ], dtype=np.float32)
    
    H_subgrid_to_world = create_homography(pts_world_subgrid, pts_subgrid)
    
    # Warp the image onto the subgrid
    projected_image_on_subgrid = warp_image_to_grid(subgrid, image, H_subgrid_to_world, H_world_to_image, subgrid_height, subgrid_width)
    
    # Finding Diff
    # Preprocess
    subgrid, projected_image_on_subgrid = preprocess_images(subgrid, projected_image_on_subgrid, applying=0)
    
    # Compute difference (only positive changes)
    diff_map_subgrid = compute_positive_difference(subgrid, projected_image_on_subgrid)
    
    # Filter mask
    diff_map_subgrid = postprocess_difference_map(diff_map_subgrid, projected_image_on_subgrid, threshold=0, temp_threshold=0)
    
    # Find Clusters
    centers, label_map = find_cluster_centers_conditional(
        diff_map=diff_map_subgrid,
        threshold=1,
        eps=20,
        min_samples=1,
        min_contrast=1
    )

    # =============================================================

    ### ================= For testing =============================
    # Calcolating the real number of cluster_centers in the grid
    H_image_to_world = create_homography(pts_world_image, pts_image)
    H_world_to_subgrid = create_homography(pts_subgrid, pts_world_subgrid)

    def apply_homography(H, point):
        """Applies a 3x3 homography H to a 2D point (x, y)."""
        x, y = point
        vec = np.array([x, y, 1.0])
        transformed = H @ vec
        return (transformed[0] / transformed[2], transformed[1] / transformed[2])
    
    grid_coords = []

    for (cx, cy) in cluster_centers:
        # Step 1: image → world
        x_world, y_world = apply_homography(H_image_to_world, (cx, cy))
    
        # Step 2: world → grid
        x_subgrid, y_subgrid = apply_homography(H_world_to_subgrid, (x_world, y_world))
    
        # Save result
        grid_coords.append((x_subgrid, y_subgrid))

    out_of_bounds = []

    for (x_subgrid, y_subgrid) in grid_coords:
        if not (0 <= x_subgrid < subgrid_width and 0 <= y_subgrid < subgrid_height):
            out_of_bounds.append((x_subgrid, y_subgrid))

    num_out = len(out_of_bounds)

    ### =======================================================

    ### ================= Plots ===============================
    # Saving fig  
    # Create a figure with 3 subplots in a single row
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    drone_pos = (x, y)
    range_xy=15000
    # === Plot 1: Drone position and camera footprint ===
    ax = axs[0]
    ax.plot(*drone_pos, 'ro', label='Drone Position')
    
    # Create closed polygon from corners
    polygon = np.vstack([corners, corners[0]])
    ax.plot(polygon[:, 0], polygon[:, 1], 'b-', label='Camera Footprint')
    ax.fill(polygon[:, 0], polygon[:, 1], color='lightblue', alpha=0.4)
    
    # Set axes limits and appearance
    ax.set_xlim(-range_xy, range_xy)
    ax.set_ylim(-range_xy, range_xy)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Drone Camera Ground Footprint")
    
    # === Plot 2: Difference map with colorbar ===
    im = axs[1].imshow(diff_map_subgrid, origin='lower', cmap='gray',
                       vmin=0, vmax=np.max(diff_map_subgrid), extent=extent)
    axs[1].set_title("Difference Map")
    axs[1].set_xlabel("Longitude (°)")
    axs[1].set_ylabel("Latitude (°)")
    
    # Add colorbar next to subplot 2
    fig.colorbar(im, ax=axs[1], shrink=0.8)
    
    # === Plot 3: Difference map with cluster centers ===
    axs[2].imshow(diff_map_subgrid, cmap='gray', origin='lower')
    for y, x in centers:
        axs[2].plot(x, y, 'ro')
    axs[2].set_title("Clusters on Diff Map")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Full path to save the figure
    filename = os.path.join("results", f"combined_plot_{i}.png")
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    
    # Close the figure to avoid displaying it
    plt.close(fig)

    ### ========================================================

    # ===================== Outputs ===========================
    num_clusters_in = num_clusters + num_spots - num_out
    if num_clusters_in == 0:
        if len(centers) ==0:
            ratio = 1
        else:
            ratio = 0
    else:
        ratio = len(centers) / num_clusters_in
    
    result = {
        'id_num': i,
        'Actual_num': num_clusters_in,
        'Detected_num': len(centers),
        'Ratio': ratio,
        'theta': theta,
        'phi': phi,
        'grid_range': [subgrid_width, subgrid_height],
        'clusters_center': [(float(x), float(y)) for (x, y) in grid_coords]
    }

    # ======================================================

    return result



# Entry point for script execution
if __name__ == "__main__":

    lon_origin, lon_max = -15000 , 15000
    lat_origin, lat_max = -15000 , 15000
    resolution = 0.25

    grid_info = {
    'lon_origin': lon_origin,
    'lon_max': lon_max,
    'lat_origin': lat_origin,
    'lat_max': lat_max,
    'resolution': resolution
    }

    
    grid_width = int((lon_max - lon_origin) / resolution)
    grid_height = int((lat_max - lat_origin) / resolution)

    original_grid  = np.zeros((grid_height, grid_width), dtype=np.uint8)+1

    results = []  # List to store results from all (phi, theta) pairs
    
    # Define PHI and corresponding repetition counts
    phi_values = [0, 8.5, 16, 25, 34, 43.5, 52.5, 61.5, 70.5, 79.5]
    repetitions = [1, 4, 7, 10, 13, 16, 18, 19, 22, 24]

    phi_values = [0, 79.5]
    repetitions = [1, 24]
    
    # Build PHI and THETA lists
    PHI = []
    THETA = []
    
    for phi, reps in zip(phi_values, repetitions):
        PHI.extend([phi] * reps)
        step = 360 / reps
        theta_values = [270 + step * i for i in range(reps)]
        THETA.extend(theta_values)
    
    # Convert to numpy arrays (optional)
    PHI = np.array(PHI)
    THETA = np.array(THETA)

    id_num = 0
    for phi, theta in zip(PHI, THETA):
        result = main(grid_info, theta=theta, phi=phi, original_grid=original_grid, i=id_num)
        if result is not None:
            results.append(result)
            id_num = id_num + 1

    # Save the DataFrame to results/results.csv
    df = pd.DataFrame(results)
    csv_path = os.path.join("results", "results.csv")
    df.to_csv(csv_path, index=False)
    print(df.head())

    






