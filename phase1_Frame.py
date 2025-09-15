import os
import pandas as pd
import matplotlib.pyplot as plt
from wildfire_detector.utils_Frame import *
import cv2
import random
import matplotlib.patches as patches
import yaml
import time

Scan0 = {}



def main(image1, Frame_index, x1, y1, h1, theta1, phi1, hfov1):
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
    times = {}
    t0 = time.time()
    
    global Scan0

    # # Fire Max Size (length)
    # fire_size = 10 # [m]
    
    # DB_Scan parameters
    min_samples_factor = config['dbscan']['min_samples_factor']
    eps_distance_factor = config['dbscan']['eps_distance_factor']
    # # Important Calculation
    # Slant_Range = h1 * 0.001 / np.cos(np.deg2rad(phi1))  # Slant range from camera to ground (meters)
    # HFOV = hfov1  # Horizontal field of view (degrees)
    # IFOV = HFOV / 1920 / 180 * np.pi * 1_000_000  # Instantaneous Field of View [urad]
    # GSD = Slant_Range * IFOV / 1000  # Ground Sampling Distance [meters per pixel]
    #
    # fire_length_pixel = np.floor(fire_size / GSD)
    # fire_num_pixel = fire_length_pixel**2

    # FOV calc for Phase 2
    # patch_length = 224  # total patch size in pixels
    # ratio_patch = 0.7         # fire ratio within the patch
    ratio_image = config['fire']['ratio_in_rgb_image']  # fire ratio within the RGB image
    IR2RGB_ratio = rgb_width / ir_width  # resolution ratio between RGB and IR images
    min_fov = config['fov']['min_deg']  # degrees - minimal allowed FOV
    max_fov = config['fov']['max_deg']  # degrees - maximal allowed FOV

    image_height = image1.shape[0]
    image_width = image1.shape[1]

    corners_0 = Scan0["corners0"][Frame_index]
    image0 = Scan0["Scan0_Images"][Frame_index]
    
    pixels_img0_at_img1, corners_1 = Reuven_Function(corners_0, Frame_index, theta1, phi1, x1, y1, h1, hfov1)

    # Step 1: Define the image corners in pixel coordinates for image1
    # Format: [top-left, top-right, bottom-right, bottom-left]
    pts_image = generate_uniform_grid(image_height, image_width, points_num=config['grid']['points_per_frame'])

    # Homography
    H_image1_to_image0 = create_homography(pts_image, pixels_img0_at_img1)
    
    # Step 5: Warp image0 into the pixel frame of image1 using the computed homography
    Image0_projected = cv2.warpPerspective(
        image0,                        # source image to warp
        H_image1_to_image0,            # transformation from image1 pixels to image0 projection
        (image1.shape[1], image1.shape[0]),  # output size (same as image1)
        cv2.INTER_LINEAR,              # interpolation method
        borderMode=cv2.BORDER_CONSTANT,           # fill borders with constant value
        borderValue=np.median(image0)             # use median of image0 for padding
    )

    # === Preprocess and Compute Difference ===

    # Step 1: Preprocess images (convert to uint8, optionally normalize)
    image1, Image0_projected = preprocess_images(image1, Image0_projected, applying=config['preprocessing']['apply'])
    
    # Step 2: Compute positive difference (changes in image1 that are brighter than image0 projection)
    diff_map = compute_positive_difference(Image0_projected, image1)
    
    # Step 3: Post-process the difference map to suppress noise and irrelevant areas
    diff_map = postprocess_difference_map(diff_map, image1, threshold=config['postprocessing']['threshold'], temp_threshold=config['postprocessing']['temp_threshold'])
    
    # === Cluster Detection Based on Difference Map ===
    
    # Step 1: Compute DBSCAN parameters based on estimated fire characteristics
    min_samples = int(np.ceil(fire_num_pixel / min_samples_factor))
    min_samples = max(1, min_samples)
    eps_distance = int(np.floor(fire_length_pixel * eps_distance_factor))
    eps_distance = max(2, eps_distance)
    
    # Step 2: Run conditional DBSCAN clustering to identify potential fire regions
    print(f"min_samples: {min_samples}")
    centers, label_map, bboxes = find_cluster_centers_conditional(
        diff_map=diff_map,
        threshold=config['dbscan']['diff_threshold'],  # Only consider pixels with diff > diff_threshold
        eps=eps_distance,       # Clustering radius
        min_samples=min_samples,  # Minimum number of points in cluster
        min_contrast=config['dbscan']['min_contrast']  # Contrast-based center selection
    )

    # Compute scores
    scores = compute_cluster_scores(label_map, image1, GSD, norm_size=config['scoring']['norm_size'],
                                    norm_intensity=config['scoring']['norm_intensity'])

    # === Compute Required FOVs Based on Detected Cluster Bounding Boxes ===
    # Initialize lists to store computed FOVs per bounding box
    required_fov2 = []  # FOVs based on entire image resolution requirement
    
    # Loop through each detected bounding box
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
    
        # Compute width and height of the bounding box in IR pixel coordinates
        width = x_max - x_min + 1 # +1 to include both endpoints (pixel indices are inclusive)
        height = y_max - y_min + 1 # +1 to include both endpoints (pixel indices are inclusive)
    
        # Take the longer dimension as the dominant fire size
        pixels_IR_at_current = max(width, height)
    
        # Convert fire size from IR pixels to RGB pixels
        pixels_RGB_at_current = pixels_IR_at_current * IR2RGB_ratio
    
        # --- FOV Type 2: Based on whole image resolution requirement ---
        fov2 = HFOV / (ratio_image * rgb_height / pixels_RGB_at_current)
        fov_clipped2 = round(np.clip(fov2, min_fov, max_fov), 2)

        required_fov2.append(fov_clipped2)

    total_time = time.time() - t0
    print(f"\n=== Inference Timing Breakdown ===")
    print(f"{'Total':>12}: {total_time*1000:.2f} ms\n")
    
    ### ================= Plots ===============================
    # Saving fig  
    # Create a figure with 3 subplots in a single row
    fig, axs = plt.subplots(1, 4, figsize=(18, 6))
    drone_pos = (x1, y1)
    range_xy=15000
    
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
    filename = os.path.join("results", f"combined_plot_{Frame_index}.png")
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    
    # Close the figure to avoid displaying it
    plt.close(fig)

    ### ========================================================


    # ===================== Outputs ===========================
    results = []
    for i in range(len(centers)):
        results.append({
            'Frame_index': Frame_index,
            'loc': centers[i],
            'bbox': bboxes[i],
            'confidence_pct': scores[i],
            'required_fov2': required_fov2[i]
        })

    # ======================================================

    return results


def test(image1, image0, x1, y1, h1, theta1, phi1, hfov1):
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
    global Scan0

    # Fire Max Size (length)
    fire_size = 10  # [m]

    # DB_Scan parameters
    min_samples_factor = 10
    eps_distance_factor = 1.5
    # Important Calculation
    # Calculations
    Slant_Range = h1 * 0.001 / np.cos(np.deg2rad(phi1))  # Slant range from camera to ground (meters)
    HFOV = hfov1  # Horizontal field of view (degrees)
    IFOV = HFOV / 1920 / 180 * np.pi * 1_000_000  # Instantaneous Field of View [urad]
    GSD = Slant_Range * IFOV / 1000  # Ground Sampling Distance [meters per pixel]

    fire_length_pixel = np.floor(fire_size / GSD)
    fire_num_pixel = fire_length_pixel ** 2

    # FOV calc for Phase 2
    patch_length = 224  # total patch size in pixels
    ratio_patch = 0.7  # fire ratio within the patch
    ratio_image = 0.25  # fire ratio within the RGB image
    IR2RGB_ratio = 1920 / 1280  # resolution ratio between RGB and IR images
    rgb_len = 1080
    min_fov = 2.2  # degrees - minimal allowed FOV
    max_fov = 60.0  # degrees - maximal allowed FOV

    image_height = image1.shape[0]
    image_width = image1.shape[1]


    # === Preprocess and Compute Difference ===

    # Step 1: Preprocess images (convert to uint8, optionally normalize)
    image0, image1= preprocess_images(image0, image1, applying=0)

    # Step 2: Compute positive difference (changes in image1 that are brighter than image0 projection)
    diff_map = compute_positive_difference(image0, image1)

    # Step 3: Post-process the difference map to suppress noise and irrelevant areas
    # threshold = 0 keeps only strictly positive differences
    # temp_threshold = 0 filters out low-intensity regions in image1
    diff_map = postprocess_difference_map(diff_map, image1, threshold=20, temp_threshold=None)

    # === Cluster Detection Based on Difference Map ===

    # Step 1: Compute DBSCAN parameters based on estimated fire characteristics
    min_samples = int(np.ceil(fire_num_pixel / min_samples_factor))
    eps_distance = int(np.floor(fire_length_pixel * eps_distance_factor))

    # Step 2: Run conditional DBSCAN clustering to identify potential fire regions
    centers, label_map, bboxes = find_cluster_centers_conditional(
        diff_map=diff_map,
        threshold=10,  # Only consider pixels with diff > 10
        eps=eps_distance,  # Clustering radius
        min_samples=min_samples,  # Minimum number of points in cluster
        min_contrast=10  # Contrast-based center selection
    )

    # Compute scores
    scores = compute_cluster_scores(label_map, image1, GSD)

    # === Compute Required FOVs Based on Detected Cluster Bounding Boxes ===

    # Estimate the desired fire size in RGB pixel scale (target size for zoom decision)
    pixels_RGB_at_patch = patch_length * ratio_patch  # target fire size in RGB pixels

    # Initialize lists to store computed FOVs per bounding box
    required_fov2 = []  # FOVs based on entire image resolution requirement

    # Loop through each detected bounding box
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox

        # Compute width and height of the bounding box in IR pixel coordinates
        width = x_max - x_min
        height = y_max - y_min

        # Take the longer dimension as the dominant fire size
        pixels_IR_at_current = max(width, height)

        # Convert fire size from IR pixels to RGB pixels
        pixels_RGB_at_current = pixels_IR_at_current * IR2RGB_ratio

        # --- FOV Type 2: Based on whole image resolution requirement ---
        fov2 = HFOV / (ratio_image * rgb_len / pixels_RGB_at_current)
        fov_clipped2 = round(np.clip(fov2, min_fov, max_fov), 2)

        required_fov2.append(fov_clipped2)

    # ==================== PLOTS ====================

    # Create figure with 4 subplots in a single row
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))

    # === Plot 0: Image 0 (Before) ===
    axs[0].imshow(image0, cmap='gray')
    axs[0].set_title("Image 0 (Before)")
    axs[0].axis('off')

    # === Plot 1: Image 1 (After) ===
    axs[1].imshow(image1, cmap='gray')
    axs[1].set_title("Image 1 (After)")
    axs[1].axis('off')

    # === Plot 2: Difference map with colorbar ===
    im = axs[2].imshow(diff_map, cmap='gray', vmin=0, vmax=np.max(diff_map))
    axs[2].set_title("Difference Map")
    axs[2].set_xlabel("Pixel X")
    axs[2].set_ylabel("Pixel Y")
    fig.colorbar(im, ax=axs[2], shrink=0.8)

    # === Plot 3: Clusters on difference map ===
    axs[3].imshow(diff_map, cmap='gray')
    axs[3].set_title("Clusters on Diff Map")
    axs[3].set_xlabel("Pixel X")
    axs[3].set_ylabel("Pixel Y")

    # Overlay red markers at detected cluster centers
    for y, x in centers:
        axs[3].plot(x, y, 'ro')

    # Overlay bounding boxes around detected clusters
    for min_y, min_x, max_y, max_x in bboxes:
        rect = patches.Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            linewidth=1.5,
            edgecolor='lime',
            facecolor='none'
        )
        axs[3].add_patch(rect)

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()

    ### ========================================================

    # ===================== Outputs ===========================
    results = []
    for i in range(len(centers)):
        results.append({
            'Frame_index': 1,
            'loc': centers[i],
            'bbox': bboxes[i],
            'confidence_pct': scores[i],
            'required_fov2': required_fov2[i]
        })

    # ======================================================

    return results


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

def Creating_Scan0(PHI, THETA, x0, y0, h0, hfov0, image0=None):
    """xxx"""
    global Scan0
    corners0 = []
    Frame_Index = {}

    # 1. Generate uniform background values
    image_height=720
    image_width=1280
    background_range = (3, 7)
    
    # Prepare image stack
    if image0 is not None:
        # Resize if needed
        image0 = cv2.resize(image0, (image_width, image_height))
        Scan0_Images = np.repeat(image0[np.newaxis, :, :], PHI.shape[0], axis=0)
    else:
        Scan0_Images = np.random.uniform(*background_range, size=(PHI.shape[0], image_height, image_width)).astype(np.uint8)
 
    i = 0
    # Loop over all (PHI, THETA) combinations and draw projected camera footprints
    for Phi, Theta in zip(PHI, THETA):
        corners0.append(pixel2geo(theta_deg=Theta, phi_deg=Phi, h=h0, x=x0, y=y0, hfov_deg=hfov0))
        Frame_Index[i] = i
        i += 1

    Scan0 = {"Scan0_Images": Scan0_Images,
             "corners0": corners0,
             "Frame_Index": Frame_Index}
    
             
def Creating_Scan1_Frame(fire_length_pixel):
    """xxx"""
    # Generate a synthetic frame (e.g., at time t1) with clustered fire-like patterns
    image1, num_clusters, cluster_centers = create_synthetic_image_with_clusters(
        image_height=720,
        image_width=1280,
        background_range=(3, 7),  # Simulate a low-intensity background
        cluster_value=200,        # Peak intensity for simulated "fire" clusters
        num_clusters_range=(1, 4),
        cluster_radius_range=(np.round(fire_length_pixel/6), np.round(fire_length_pixel/2))
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


def Reuven_Function(corners_0, Frame_index, theta1, phi1, x1=0, y1=7500, h1=2500, hfov1=17.5):
    """xxx"""

    image_height = 720
    image_width = 1280
    
    
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
    H_world_to_image1 = create_homography(pts_image, corners_1)
    
    # Step 3: Project the world-space corners of image0 (corners_0) into image1's pixel space
    pixels_img0_at_img1 = project_points_with_homography(corners_0, H_world_to_image1)

    return pixels_img0_at_img1, corners_1




# Entry point for script execution - Original Scenario
if __name__ == "__main__":

    with open("wildfire_detector'/config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # Parameters
    x0, y0 = 0, 7500
    h0 = 2500
    theta0 = 0 # yaw
    phi0 = 0   # pitch
    hfov0 = 17.5   # horizontal field of view

    # Define standard deviations (σ) for each parameter
    x_std = 2
    y_std = 2
    h_std = 2
    theta_std = 1
    phi_std = 0.2
    hfov_std = 0.1

    # Fire Max Size (length)
    fire_size = config['fire']['max_size_m'] # [m]

    results = []  # List to store results from all (phi, theta) pairs

    # Defining PHI, THETA
    PHI, THETA = Phi_Theta_Generation()

    # Creating scan_0
    Creating_Scan0(PHI, THETA, x0, y0, h0, hfov0)

    Frame_index = 0
    for phi, theta in zip(PHI, THETA):
        # Generate Scan 1 values using Gaussian distribution (mean=0)
        x1 = x0 + random.gauss(0, x_std)
        y1 = y0 + random.gauss(0, y_std)
        h1 = h0 + random.gauss(0, h_std)
        theta1 = THETA[Frame_index] + random.gauss(0, theta_std)
        phi1 = PHI[Frame_index] + random.gauss(0, phi_std)
        hfov1 = hfov0 + random.gauss(0, hfov_std)

        # Important Calculation
        rgb_height, rgb_width = config['image']['rgb_size']  # [width, height]
        ir_height, ir_width = config['image']['ir_size']
        Slant_Range = h1 * 0.001 / np.cos(np.deg2rad(phi1))  # Slant range from camera to ground (meters)
        HFOV = hfov1  # Horizontal field of view (degrees)
        IFOV = HFOV / rgb_width / 180 * np.pi * 1_000_000  # Instantaneous Field of View [urad]
        GSD = Slant_Range * IFOV / 1000  # Ground Sampling Distance [meters per pixel]

        fire_length_pixel = np.floor(fire_size / GSD)
        fire_num_pixel = fire_length_pixel**2

        # Creating frame 1 (scan 1)
        image1, clusters_num, cluster_centers = Creating_Scan1_Frame(fire_length_pixel)

        result = main(image1, Frame_index, x1, y1, h1, theta1, phi1, hfov1)

        # Compute performance comparison
        # num_out = 0  # or however you're computing removed clusters
        num_clusters_in = clusters_num #- num_out
        detected = len(result) if result is not None else 0
        ratio = detected / num_clusters_in if num_clusters_in > 0 else (1 if detected == 0 else 0)

        if result is not None:
            # Attach evaluation values to each detection
            for r in result:
                r["ground_truth"] = num_clusters_in
                r["detected"] = detected
                r["detection_ratio"] = ratio
            results.extend(result)

        Frame_index += 1

    # Save the DataFrame to results/results.csv
    df = pd.DataFrame(results)
    csv_path = os.path.join("results", "results.csv")
    df.to_csv(csv_path, index=False)
    print(df.head())


# # Entry point for script execution - Image Comparison
# if __name__ == "__main__":
#
#     # Parameters
#     x0, y0 = 0, 7500
#     h0 = 2500
#     theta0 = 0 # yaw
#     phi0 = 0   # pitch
#     hfov0 = 17.5   # horizontal field of view
#
#     # Define standard deviations (σ) for each parameter
#     x_std = 2
#     y_std = 2
#     h_std = 2
#     theta_std = 1
#     phi_std = 0.2
#     hfov_std = 0.1
#
#     # Fire Max Size (length)
#     fire_size = 10 # [m]
#
#     results = []  # List to store results from all (phi, theta) pairs
#
#     # Defining PHI, THETA
#     PHI, THETA = np.array([0]), np.array([0])
#
#     # Creating scan_0
#
#     # Image0 - Load the thermal image as-is
#     # image0 = cv2.imread(r'HeronFlight\extracted_frames\ir_frames\frame_6793.jpg', cv2.IMREAD_UNCHANGED)
#     # # Option 1: Convert to grayscale (standard luminance formula)
#     # image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
#     # # Option 2 (alternative): Use only one channel (if it's duplicated)
#     # # image_gray = image_color[:, :, 2]  # e.g., Red channel
#     # # Normalize to 0–255
#     # image0 = cv2.normalize(image0, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#     # image0 = image0.astype(np.uint8)
#
#     # Path to your CSV file
#     csv_path = 'dataset_FOR_IAI/tc004_1.csv'
#
#     # Load the CSV, skip the first 4 rows and first column (column A)
#     df = pd.read_csv(csv_path, header=None, skiprows=4, usecols=lambda col: col not in [0])
#
#     # Optional: Reset column indices to numeric if needed
#     df.columns = range(df.shape[1])
#
#     # Convert to numpy array with float32 precision
#     image_array = df.to_numpy(dtype=np.float32)
#     image_array = image_array[:,1:]
#     image0 = image_array.astype(np.uint8)
#     # Resize if needed
#     image0 = cv2.resize(image0, (1280, 720))
#
#     # # Image1 - Load the thermal image as-is
#     # image1 = cv2.imread(r'HeronFlight\extracted_frames\ir_frames\frame_6796.jpg', cv2.IMREAD_UNCHANGED)
#     # # Option 1: Convert to grayscale (standard luminance formula)
#     # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     # # Option 2 (alternative): Use only one channel (if it's duplicated)
#     # # image_gray = image_color[:, :, 2]  # e.g., Red channel
#     # # Resize if needed
#     # image1 = cv2.resize(image1, (1280, 720))
#     # # Normalize to 0–255
#     # image1 = cv2.normalize(image1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#     # image1 = image1.astype(np.uint8)
#
#     # Path to your CSV file
#     csv_path = 'dataset_FOR_IAI/tc004_3.csv'
#
#     # Load the CSV, skip the first 4 rows and first column (column A)
#     df = pd.read_csv(csv_path, header=None, skiprows=4, usecols=lambda col: col not in [0])
#
#     # Optional: Reset column indices to numeric if needed
#     df.columns = range(df.shape[1])
#
#     # Convert to numpy array with float32 precision
#     image_array = df.to_numpy(dtype=np.float32)
#     image_array = image_array[:,1:]
#     image1 = image_array.astype(np.uint8)
#     # Resize if needed
#     image1 = cv2.resize(image1, (1280, 720))
#
#     Creating_Scan0(PHI, THETA, x0, y0, h0, hfov0, image0)
#
#     #Plotting the images
#     plt.figure(figsize=(10, 5))
#     # Plot image0
#     plt.subplot(1, 2, 1)
#     cmap0 = 'gray' if len(image0.shape) == 2 else None
#     plt.imshow(image0, cmap=cmap0)
#     plt.title('Image 0')
#     plt.axis('off')
#
#     # Plot image1
#     plt.subplot(1, 2, 2)
#     cmap1 = 'gray' if len(image1.shape) == 2 else None
#     plt.imshow(image1, cmap=cmap1)
#     plt.title('Image 1')
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#
#     Frame_index = 0
#     for phi, theta in zip(PHI, THETA):
#         # Generate Scan 1 values using Gaussian distribution (mean=0)
#         x1 = x0 + random.gauss(0, x_std)
#         y1 = y0 + random.gauss(0, y_std)
#         h1 = h0 + random.gauss(0, h_std)
#         theta1 = THETA[Frame_index] + random.gauss(0, theta_std)
#         phi1 = PHI[Frame_index] + random.gauss(0, phi_std)
#         hfov1 = hfov0 + random.gauss(0, hfov_std)
#
#         # Important Calculation
#         # Calculations
#         Slant_Range = h1 * 0.001 / np.cos(np.deg2rad(phi1))  # Slant range from camera to ground (meters)
#         HFOV = hfov1  # Horizontal field of view (degrees)
#         IFOV = HFOV / 1920 / 180 * np.pi * 1_000_000  # Instantaneous Field of View [urad]
#         GSD = Slant_Range * IFOV / 1000  # Ground Sampling Distance [meters per pixel]
#
#         fire_length_pixel = np.floor(fire_size / GSD)
#         fire_num_pixel = fire_length_pixel**2
#         print(f'fire size in pixels {fire_num_pixel}')
#
#
#         result = main(image1, Frame_index, x1, y1, h1, theta1, phi1, hfov1)
#
#         # # Compute performance comparison
#         # num_out = 0  # or however you're computing removed clusters
#         # num_clusters_in = clusters_num - num_out
#         # detected = len(result) if result is not None else 0
#         # ratio = detected / num_clusters_in if num_clusters_in > 0 else (1 if detected == 0 else 0)
#
#         if result is not None:
#             # Attach evaluation values to each detection
#             # for r in result:
#                 # r["ground_truth"] = num_clusters_in
#                 # r["detected"] = detected
#                 # r["detection_ratio"] = ratio
#             results.extend(result)
#
#         Frame_index += 1
#
#     # Save the DataFrame to results/results.csv
#     df = pd.DataFrame(results)
#     csv_path = os.path.join("results", "results.csv")
#     df.to_csv(csv_path, index=False)
#     print(df.head())

# if __name__ == "__main__":

#     # Parameters
#     x1, y1 = 0, 7500
#     h1 = 2500
#     theta1 = 45  # yaw
#     phi1 = 0 # pitch
#     hfov1= 17.5  # horizontal field of view

#     # Load the thermal image as-is
#     image0 = cv2.imread(r'HeronFlight\extracted_frames\ir_frames\frame_6793.jpg', cv2.IMREAD_UNCHANGED)
#     # Option 1: Convert to grayscale (standard luminance formula)
#     image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
#     # Option 2 (alternative): Use only one channel (if it's duplicated)
#     # image_gray = image_color[:, :, 2]  # e.g., Red channel
#     # Normalize to 0–255
#     image0 = cv2.normalize(image0, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#     image0 = image0.astype(np.uint8)


#     # Load the thermal image as-is
#     image1 = cv2.imread(r'HeronFlight\extracted_frames\ir_frames\frame_6793.jpg', cv2.IMREAD_UNCHANGED)
#     # Option 1: Convert to grayscale (standard luminance formula)
#     image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     # Option 2 (alternative): Use only one channel (if it's duplicated)
#     # image_gray = image_color[:, :, 2]  # e.g., Red channel
#     # Normalize to 0–255
#     image1 = cv2.normalize(image1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#     image1 = image1.astype(np.uint8)

#     print(image1.shape)

#     import matplotlib

#     matplotlib.use('TkAgg')  # Force external window for plots
#     import matplotlib.pyplot as plt

#     # Create histograms for image0 and image1
#     plt.figure(figsize=(12, 5))

#     # Histogram for image0
#     plt.subplot(1, 2, 1)
#     plt.hist(image0.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
#     plt.title("Histogram of Image 0")
#     plt.xlabel("Pixel Intensity")
#     plt.ylabel("Frequency")

#     # Histogram for image1
#     plt.subplot(1, 2, 2)
#     plt.hist(image1.ravel(), bins=256, range=(0, 255), color='green', alpha=0.7)
#     plt.title("Histogram of Image 1")
#     plt.xlabel("Pixel Intensity")
#     plt.ylabel("Frequency")

#     plt.tight_layout()

#     # Calculate and print total number of pixels (sum of histogram bins)
#     hist_vals, _ = np.histogram(image1, bins=256, range=(0, 255))
#     print("Total pixel count from histogram (image1):", np.sum(hist_vals))

#     plt.show()



