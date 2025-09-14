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
    H_world_to_image1 = create_homography(pts_image, corners_1)
    
    # Step 3: Project the world-space corners of image0 (corners_0) into image1's pixel space
    pixels_img0_at_img1 = project_points_with_homography(corners_0, H_world_to_image1)

    return pixels_img0_at_img1, corners_1


