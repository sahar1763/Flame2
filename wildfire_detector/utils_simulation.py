import os
import pandas as pd
import matplotlib.pyplot as plt
from wildfire_detector.utils_Frame import *
import cv2
import random
import matplotlib.patches as patches
import yaml
import time


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


