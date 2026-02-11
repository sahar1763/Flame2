from wildfire_detector.function_class_demo import ScanManager
import time

import os
import pandas as pd
import matplotlib.pyplot as plt
from wildfire_detector.utils_Frame import *
from wildfire_detector.utils_simulation import *
import cv2
import random
import matplotlib.patches as patches
import yaml
import time
import copy


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
    phase1_inputs_imgs, phase1_inputs_metadata, clusters_num_array, _ = Creating_Phase1_input(PHI, THETA, x0, y0, h0, hfov0, metadata, sm.config)

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
    metadata["investigation_parameters"]["detected_bounding_box"] = [540-140, 960-140, 0, 540+140, 960+140, 0]

    print("\033[1;96m=== Starting phase2 ===\033[0m")
    for i in range(10):
        print(f"\n{i}\n")
        result_phase2 = sm.phase2(frame_rgb1, metadata)

    print("OK")




