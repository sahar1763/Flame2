from wildfire_detector.function_class_demo import ScanManager
import time

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
        cluster_radius_range=(np.round(fire_length_pixel / 6), np.round(fire_length_pixel / 2))
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

def Creating_Phase1_input(PHI, THETA, h0, hfov0, metadata, config):
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
        rgb_height, rgb_width = config['image']['rgb_size']  # [width, height]
        ir_height, ir_width = config['image']['ir_size']
        Slant_Range = h1 * 0.001 / np.cos(np.deg2rad(phi1))  # Slant range from camera to ground (meters)
        HFOV = hfov1  # Horizontal field of view (degrees)
        IFOV = HFOV / rgb_width / 180 * np.pi * 1_000_000  # Instantaneous Field of View [urad]
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


if __name__ == "__main__":

    # === Init ScanManager ===
    sm = ScanManager()

    # === Create dummy metadata ===
    metadata = sm.dummy_md

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

    # Defining PHI, THETA
    PHI, THETA = Phi_Theta_Generation()

    # ==== Phase 0 ====

    # Creating scan_0_inputs
    scan_0_inputs_imgs, scan_0_inputs_metadata = Creating_Scan0_input(PHI, THETA, h0, hfov0, metadata, sm.config)

    for i in range(scan_0_inputs_imgs.shape[0]):
        frame_ir = scan_0_inputs_imgs[i]
        metadata = scan_0_inputs_metadata[i]
        sm.phase0(frame_ir, metadata)

    # ==== Phase 1 ====

    # Creating phase1_inputs
    phase1_inputs_imgs, phase1_inputs_metadata, clusters_num_array, _ = Creating_Phase1_input(PHI, THETA, h0, hfov0, metadata, sm.config)

    # # Fire Max Size (length)
    # fire_size = sm.config['fire']['max_size_m']  # [m]

    results = []  # List to store results from all (phi, theta) pairs


    for i in range(len(phase1_inputs_imgs)):
        frame_ir = phase1_inputs_imgs[i]
        metadata = phase1_inputs_metadata[i]
        clusters_num = clusters_num_array[i]

        tt0 = time.perf_counter()
        result = sm.phase1(frame_ir, metadata)
        total_time = time.perf_counter() - tt0
        print(f"\n=== Phase1 Total Runtime === {total_time * 1000:.2f} msec\n")

        detected = len(result) if result is not None else 0
        ratio = detected / clusters_num if clusters_num > 0 else (1 if detected == 0 else 0)

        if result is not None:
            # Attach evaluation values to each detection
            for r in result:
                r["ground_truth"] = clusters_num
                r["detected"] = detected
                r["detection_ratio"] = ratio
            results.extend(result)


    # Save the DataFrame to results_demoPackage/results.csv
    df = pd.DataFrame(results)
    results_dir = "results_demoPackage"
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(df.head())

    # ==== Phase 2 ====

    # Loading RGB image
    rgb_size = sm.config["image"]["rgb_size"] # [height, width] = [1080, 1920]
    image_path = r"C:/Projects/Flame2/Datasets_FromDvir/Datasets/rgb_images/00216-fire-rgb-flame3.JPG"
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Step 1: Resize image to 600x600
    resized = cv2.resize(image_rgb, (600, 600), interpolation=cv2.INTER_LINEAR)

    # Step 2: Create a black canvas (1080x1920)
    frame_rgb1 = np.zeros((rgb_size[0], rgb_size[1], 3), dtype=np.uint8)

    # Step 3: Compute top-left corner to paste the resized image in the center
    y_offset = (rgb_size[0] - 600) // 2
    x_offset = (rgb_size[1] - 600) // 2

    # Step 4: Paste resized image into the center of the canvas
    frame_rgb1[y_offset:y_offset + 600, x_offset:x_offset + 600] = resized

    # Creating metadata for phase2:
    metadata = copy.deepcopy(sm.dummy_md)
    metadata["investigation_parameters"]["detected_bounding_box"] = [960-140, 540-140, 960+140, 540+140]

    print("\033[1;96m=== Starting phase2 ===\033[0m")
    for i in range(10):
        print(f"\n{i}\n")
        result_phase2 = sm.phase2(frame_rgb1, metadata)

    print("OK")




