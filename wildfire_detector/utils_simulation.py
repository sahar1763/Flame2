import os
import pandas as pd
import matplotlib.pyplot as plt
from wildfire_detector.utils_Frame import *
import cv2
import random
import matplotlib.patches as patches
import yaml
import time
import copy


def geo2pixel(corners_0, theta1, phi1, h1=2500, x1=0, y1=7500, hfov1=17.5, img_size=[720, 1280]):
    """xxx"""

    image_height = img_size[0]
    image_width = img_size[1]

    # Compute ground-intersection corners for the second camera position
    corners_1 = pixel2geo(theta_deg=theta1, phi_deg=phi1, h=h1, x=x1, y=y1, hfov_deg=hfov1)

    # Step 1: Define the image corners in pixel coordinates for image1
    # Format: [top-left, top-right, bottom-right, bottom-left]
    pts_image = generate_uniform_grid(image_height, image_width, points_num=4)
    # pts_image = np.array([
    #       [0, 0],                                # top-left
    #       [image_width - 1, 0],                  # top-right
    #       [image_width - 1, image_height - 1],   # bottom-right
    #       [0, image_height - 1]                  # bottom-left
    #   ], dtype=np.float32)
    
    # Step 2: Compute homography that maps world coordinates (corners_1) to image1 pixels
    H_world_to_image1 = create_homography(pts_image, corners_1[:,:2])
    
    # Step 3: Project the world-space corners of image0 (corners_0) into image1's pixel space
    pixels_img0_at_img1 = project_points_with_homography(corners_0[:,:2], H_world_to_image1)

    return pixels_img0_at_img1, corners_1


def plot_phase1(diff_map, corners_0, corners_1, centers, bboxes, Frame_index, x1=0, y1=7500):
    ### ================= Plots ===============================
    # Saving fig
    # Create a figure with 3 subplots in a single row
    fig, axs = plt.subplots(1, 4, figsize=(18, 6))
    drone_pos = (x1, y1)
    range_xy = 15000

    # === Plot 1: Drone position and camera footprint ===
    ax = axs[0]
    ax.plot(*drone_pos, 'ro', label='Drone Position')

    # Create closed polygon from corners
    polygon1 = np.vstack([corners_0[0], corners_0[1], corners_0[3], corners_0[2], corners_0[0]])
    ax.plot(polygon1[:, 0], polygon1[:, 1], 'b-', label='Camera Footprint 0')
    ax.fill(polygon1[:, 0], polygon1[:, 1], color='lightblue', alpha=0.4)
    polygon2 = np.vstack([corners_1[0], corners_1[1], corners_1[3], corners_1[2], corners_1[0]])
    ax.plot(polygon2[:, 0], polygon2[:, 1], 'g-', label='Camera Footprint 1')
    ax.fill(polygon2[:, 0], polygon2[:, 1], color='lightgreen', alpha=0.4)

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
    im = axs[1].imshow(diff_map, cmap='gray', vmin=0, vmax=np.max(diff_map))
    axs[1].set_title("Difference Map")
    axs[1].set_xlabel("Pixel X")
    axs[1].set_ylabel("Pixel Y")

    # Add colorbar next to subplot 2
    fig.colorbar(im, ax=axs[1], shrink=0.8)

    # === Plot 3: Difference map with cluster centers ===
    axs[2].imshow(diff_map, cmap='gray')
    # Overlay red markers at detected cluster centers
    for y, x in centers:
        axs[2].plot(x, y, 'ro')
    axs[2].set_title("Clusters on Diff Map")
    # Overlay bounding boxes around detected clusters
    for min_y, min_x, max_y, max_x in bboxes:
        width = max_x - min_x
        height = max_y - min_y
        rect = patches.Rectangle(
            (min_x, min_y),
            width,
            height,
            linewidth=1.5,
            edgecolor='lime',
            facecolor='none'
        )
        axs[2].add_patch(rect)

    # === Plot 4: Camera footprint ===
    ax = axs[3]
    ax.plot(*drone_pos, 'ro', label='Drone Position')

    # Create closed polygon from corners
    ax.plot(polygon1[:, 0], polygon1[:, 1], 'b-', label='Camera Footprint 0')
    ax.fill(polygon1[:, 0], polygon1[:, 1], color='lightblue', alpha=0.4)
    ax.plot(polygon2[:, 0], polygon2[:, 1], 'g-', label='Camera Footprint 1')
    ax.fill(polygon2[:, 0], polygon2[:, 1], color='lightgreen', alpha=0.4)

    # Set axes limits and appearance
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Drone Camera Ground Footprint")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Full path to save the figure
    filename = os.path.join("results_demoPackage", f"combined_plot_{Frame_index}.png")

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    # Close the figure to avoid displaying it
    plt.close(fig)

    ### ========================================================


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


def pixel2geo(theta_deg, phi_deg, h=2500, x=0, y=7500, hfov_deg=17.5, img_size=[720,1280]):

    height = img_size[0]
    width = img_size[1]

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
        corners_world.append(point)
    return np.array(corners_world)


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
        # 3. Random cluster center (cy, cx)
        cy = np.random.randint(0, image_height)
        cx = np.random.randint(0, image_width)

        # Save the center
        cluster_centers.append((cy, cx))

        # 4. Random radius for the Gaussian cluster
        radius = np.random.randint(cluster_radius_range[0], cluster_radius_range[1] + 1)
        radius = max(radius, 1)

        # 5. Compute 2D Gaussian mask centered at (cx, cy)
        y, x = np.meshgrid(np.arange(image_height), np.arange(image_width), indexing='ij')
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        cluster_value_temp = np.random.randint(cluster_value - 50, cluster_value + 50)
        gaussian_blob = cluster_value_temp * np.exp(-dist_sq / (4 * (radius ** 2)))

        # 6. Update the image with the cluster (take maximum where overlapping)
        image = np.maximum(image, gaussian_blob).astype(np.uint8)

    return image, num_clusters, cluster_centers


# def create_synthetic_image_with_clusters(
#         image_height, image_width,
#         background_range=(3, 7),
#         cluster_value=200,
#         num_clusters_range=(1, 4),
#         cluster_radius_range=(1, 9)):
#     """
#     Creates a synthetic image with uniform random background and uniform circular clusters.
#     """
#     # 1. Uniform background
#     image = np.random.uniform(*background_range, size=(image_height, image_width)).astype(np.uint8)
#
#     # 2. Number of clusters
#     num_clusters = np.random.randint(num_clusters_range[0], num_clusters_range[1] + 1)
#     cluster_centers = []
#
#     # Coordinate grid for distance calculations
#     y_grid, x_grid = np.meshgrid(np.arange(image_height), np.arange(image_width), indexing='ij')
#
#     for _ in range(num_clusters):
#         # 3. Random center
#         cx = np.random.randint(0, image_width)
#         cy = np.random.randint(0, image_height)
#         cluster_centers.append((cx, cy))
#
#         # 4. Random radius
#         radius = np.random.randint(cluster_radius_range[0], cluster_radius_range[1] + 1)
#
#         # 5. Random cluster intensity
#         cluster_value_temp = np.random.randint(cluster_value - 50, cluster_value + 50)
#
#         # 6. Create mask for circle
#         dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
#         mask = dist_sq <= radius ** 2
#
#         # 7. Apply uniform value
#         image[mask] = cluster_value_temp
#
#     return image, num_clusters, cluster_centers


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

    # Create meshgrid in size of image dimensions
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    for _ in range(num_spots):
        # Random center
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)

        # Random radius
        radius = np.random.randint(*spot_radius_range)

        # Build mask of circular region
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        mask = dist_sq <= radius ** 2

        # Add uniform random values in the specified range to that region
        patch = np.random.uniform(*value_range, size=image.shape)
        image[mask] = patch[mask]

    return num_spots


# ==== Used at "TestingPackageCode" ====
def Phi_Theta_Generation():
    """xxx"""
    # Define PHI and corresponding repetition counts
    phi_values = [0, 8.5, 16, 25, 34, 43.5, 52.5, 61.5, 70.5, 79.5]
    repetitions = [1, 4, 7, 10, 13, 16, 18, 19, 22, 24]
    # phi_values = [0, 8.5]
    # repetitions = [1, 4]

    # Build PHI and THETA lists
    PHI = []
    THETA = []

    for phi, reps in zip(phi_values, repetitions):
        PHI.extend([phi] * reps)
        step = 360 / reps
        theta_values = [step * i for i in range(reps)]
        THETA.extend(theta_values)

    # Convert to numpy arrays (optional)
    PHI = np.array(PHI)
    THETA = np.array(THETA)

    return PHI, THETA

def Creating_Scan0_input(PHI, THETA, h0, hfov0, metadata, config):
    metadata_array = []
    # 1. Generate uniform background values
    image_height, image_width = config['image']['ir_size']
    background_range = (3, 7)

    # Prepare image stack
    Scan0_Images = np.random.uniform(*background_range, size=(PHI.shape[0], image_height, image_width)).astype(np.uint8)

    i = 0
    # Loop over all (PHI, THETA) combinations and draw projected camera footprints
    for Phi, Theta in zip(PHI, THETA):
        metadata_updated = copy.deepcopy(metadata)
        metadata_updated["uav"]["pitch_deg"] = Phi
        metadata_updated["uav"]["yaw_deg"] = Theta
        metadata_updated["uav"]["altitude_agl_meters"] = h0
        metadata_updated["payload"]["field_of_view_deg"] = hfov0
        metadata_updated["scan_parameters"]["current_scanned_frame_id"] = i
        metadata_array.append(metadata_updated)
        i += 1

    return Scan0_Images, metadata_array

def Creating_Scan1_Frame(fire_length_pixel, image_size):
    """xxx"""
    # Generate a synthetic frame (e.g., at time t1) with clustered fire-like patterns
    image_height, image_width = image_size
    image1, num_clusters, cluster_centers = create_synthetic_image_with_clusters(
        image_height=image_height,
        image_width=image_width,
        background_range=(3, 7),  # Simulate a low-intensity background
        cluster_value=200,  # Peak intensity for simulated "fire" clusters
        num_clusters_range=(1, 4),
        cluster_radius_range=(int(np.round(fire_length_pixel / 6)), int(np.round(fire_length_pixel / 2)))
    )

    # # Add uniformly noisy spots to simulate false detections or thermal clutter
    # num_spots = add_uniform_spots(
    #     image1,
    #     value_range=(50, 100),  # Intensity range for noise spots
    #     spot_radius_range=(np.round(fire_length_pixel/2), fire_length_pixel*4),
    #     num_spots_range=(2, 5)
    # )
    num_spots = 0

    clusters_num = num_clusters + num_spots

    return image1, clusters_num, cluster_centers

def Creating_Phase1_input(PHI, THETA, x0, y0, h0, hfov0, metadata, config):
    metadata_array = []
    image_array = []
    clusters_num_array = []
    cluster_centers_array = []

    # Define standard deviations (σ) for each parameter
    x_std = 2
    y_std = 2
    h_std = 2
    theta_std = 1
    phi_std = 0.2
    hfov_std = 0.1

    # Fire Max Size (length)
    fire_size = config['fire']['max_size_m']  # [m]

    i = 0
    for phi, theta in zip(PHI, THETA):
        # Generate Scan 1 values using Gaussian distribution (mean=0)
        x1 = x0 + random.gauss(0, x_std)
        y1 = y0 + random.gauss(0, y_std)
        h1 = h0 + random.gauss(0, h_std)
        theta1 = theta + random.gauss(0, theta_std)
        phi1 = phi + random.gauss(0, phi_std)
        hfov1 = hfov0 + random.gauss(0, hfov_std)

        # Important Calculation
        rgb_height, rgb_width = config['image']['rgb_size']  # [height, width]
        ir_height, ir_width = config['image']['ir_size']
        Slant_Range = h1 * 0.001 / np.cos(np.deg2rad(phi1))  # Slant range from camera to ground (meters)
        HFOV = hfov1  # Horizontal field of view (degrees)
        IFOV = HFOV / ir_width / 180 * np.pi * 1_000_000  # Instantaneous Field of View [urad]
        GSD = Slant_Range * IFOV / 1000  # Ground Sampling Distance [meters per pixel]

        fire_length_pixel = np.floor(fire_size / GSD)
        fire_num_pixel = fire_length_pixel**2

        # Creating frame 1 (scan 1)
        image1, clusters_num, cluster_centers = Creating_Scan1_Frame(fire_length_pixel, config['image']['ir_size'])
        metadata_updated = copy.deepcopy(metadata)
        metadata_updated["uav"]["pitch_deg"] = phi1
        metadata_updated["uav"]["yaw_deg"] = theta1
        metadata_updated["uav"]["altitude_agl_meters"] = h1
        metadata_updated["payload"]["field_of_view_deg"] = hfov1
        metadata_updated["scan_parameters"]["current_scanned_frame_id"] = i

        metadata_array.append(metadata_updated)
        image_array.append(image1)
        clusters_num_array.append(clusters_num)
        cluster_centers_array.append(cluster_centers)

        i = i + 1

    return image_array, metadata_array, clusters_num_array, cluster_centers_array


