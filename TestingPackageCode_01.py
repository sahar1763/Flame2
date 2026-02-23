import numpy as np
import cv2
import matplotlib.pyplot as plt
from wildfire_detector.function_class_demo import ScanManager
import time

# ============================================================
# === Init ScanManager
# ============================================================
sm = ScanManager()

# ============================================================
# === Create dummy IR images
# ============================================================
ir_height, ir_width = sm.config['image']['ir_size']

# Background noise
frame_ir_base = np.random.randint(0, 9, (ir_height, ir_width), dtype=np.uint8)

# IMPORTANT: use copy()
frame_ir_phase0 = frame_ir_base.copy()
frame_ir_phase1 = frame_ir_base.copy()

# ------------------------------------------------------------
# Insert fire block in Phase 0
# ------------------------------------------------------------
y0, x0 = 200, 300
h, w = 20, 20

frame_ir_phase0[y0:y0+h, x0:x0+w] = 255

# ------------------------------------------------------------
# Shift fire for Phase 1
# ------------------------------------------------------------
delta_y = 3
delta_x = -4

frame_ir_phase1[
    y0+delta_y : y0+delta_y+h,
    x0+delta_x : x0+delta_x+w
] = 255

# ============================================================
# === Plot images side by side
# ============================================================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Phase 0")
plt.imshow(frame_ir_phase0, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Phase 1")
plt.imshow(frame_ir_phase1, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# === Create dummy metadata
# ============================================================
metadata = {
    "uav": {
        "altitude_agl_meters": 2400.0,
        "roll_deg": 0,
        "pitch_deg": 0,
        "yaw_deg": 0,
    },
    "payload": {
        "elevation_deg": -90,
        "azimuth_deg": 0,
        "field_of_view_deg": 17.5,
        "resolution_px": [1280, 720],
    },
    "geolocation": {
        "transformation_matrix": np.eye(4, dtype=float).ravel(order="C").tolist(),
        "latitude": 31.0461,
        "longitude": 34.8516,
    },
    "investigation_parameters": {
        "detection_latitude": 31.0421,
        "detection_longitude": 34.8516,
        "detection_altitude": 0.0,
        "detected_bounding_box": [31.1, 34.8, 0.0, 31.0, 34.9, 0.0],
    },
    "scan_parameters": {
        "current_scanned_frame_id": 35,
        "total_scanned_frames": 173,
    },
    "timestamp": "2025-04-08T12:30:45.123Z",
}

# ============================================================
# === Phase 0
# ============================================================
sm.phase0(frame_ir_phase0, metadata)

# ============================================================
# === Phase 1
# ============================================================
results_phase1 = sm.phase1(frame_ir_phase1, metadata)

print("\nPhase 1 results:")
for res in results_phase1:
    print(res)