import numpy as np
import cv2
from wildfire_detector.function_class_demo import ScanManager
import time

# === Init ScanManager ===
sm = ScanManager()

# === Create dummy IR image ===
ir_height, ir_width = sm.config['image']['ir_size']
frame_ir = np.random.randint(0, 255, (ir_height, ir_width), dtype=np.uint8)

# ==== IMG 1 ====
image_path = r"C:/Projects/Flame2/Datasets_FromDvir/Datasets/rgb_images/00216-fire-rgb-flame3.JPG"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Step 1: Resize image to 600x600
resized = cv2.resize(image_rgb, (600, 600), interpolation=cv2.INTER_LINEAR)

# Step 2: Create a black canvas (1080x1920)
frame_rgb1 = np.zeros((1080, 1920, 3), dtype=np.uint8)

# Step 3: Compute top-left corner to paste the resized image in the center
y_offset = (1080 - 600) // 2
x_offset = (1920 - 600) // 2

# Step 4: Paste resized image into the center of the canvas
frame_rgb1[y_offset:y_offset + 600, x_offset:x_offset + 600] = resized

# ==== IMG 2 ====
image_path = r"C:/Projects/Flame2/Datasets_FromDvir/Datasets/rgb_images/00039-fire-rgb-flame3.JPG"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Step 1: Resize image to 600x600
resized = cv2.resize(image_rgb, (600, 600), interpolation=cv2.INTER_LINEAR)

# Step 2: Create a black canvas (1080x1920)
frame_rgb2 = np.zeros((1080, 1920, 3), dtype=np.uint8)

# Step 3: Compute top-left corner to paste the resized image in the center
y_offset = (1080 - 600) // 2
x_offset = (1920 - 600) // 2

# Step 4: Paste resized image into the center of the canvas
frame_rgb2[y_offset:y_offset + 600, x_offset:x_offset + 600] = resized

# === Create dummy metadata ===
metadata = {
    "uav": {
        "altitude_agl_meters": 2400.0,
        "roll_deg": 0.5,
        "pitch_deg": -1.2,
        "yaw_deg": 45.0,
    },
    "payload": {
        "pitch_deg ": -12.0,
        "azimuth_deg ": 128.0,
        "field_of_view_deg ": 2.5,
        "resolution_px": [1920, 1080],
    },
    "geolocation": {
        "latitude": 31.0461,
        "transformation_matrix": np.eye(4).tolist(),
        "longitude": 34.8516,
    },
    "investigation_parameters": {
        "detection_latitude": 31.0421,
        "detection_longitude ": 34.8516,
        "detected_bounding_box ": [31.1, 34.8, 31.0, 34.9]
    },
    "scan_parameters": {
        "current_scanned_frame_id": 35,
        "total_scanned_frames": 173,
    },
    "timestamp": "2025-04-08T12:30:45.123Z",  # ISO 8601 format
}
# === Phase 0: Save reference frame and corner projection ===
sm.phase0(frame_ir, metadata)

# === Phase 1: IR fire detection ===
results_phase1 = sm.phase1(frame_ir, metadata)
print("\nPhase 1 results:")
for res in results_phase1:
    print(res)

# === Phase 2: RGB fire confirmation ===
# Use a dummy bbox (normally comes from phase1)
dummy_bbox = [32.1, 34.8, 32.0, 34.9]  # [top_lat, top_lon, bottom_lat, bottom_lon]
for i in range(3):
    print(f"\n{i}\n")
    result_phase2 = sm.phase2(frame_rgb1, dummy_bbox, metadata)

# print("Phase 1")
# results_phase1 = sm.phase1(frame_ir, metadata)
# results_phase1 = sm.phase1(frame_ir, metadata)
# print("End of phase 1. Starting Phase 2.")
#
# for i in range(10):
#     if i % 2 == 0:
#         print("frame_rgb1")
#         result_phase2 = sm.phase2(frame_rgb1, dummy_bbox, metadata)
#     else:
#         print("frame_rgb2")
#         result_phase2 = sm.phase2(frame_rgb2, dummy_bbox, metadata)

# print("\nPhase 2 result:")
# print(result_phase2)