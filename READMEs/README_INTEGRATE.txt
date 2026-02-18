### For running the demo package:
# In order to run demo, Instead of:
# from wildfire_detector.function_class_TensorRT import ScanManager
# Insert:
# from wildfire_detector.function_class_demo_TensorRT import ScanManager
# And instead of (in "metadata"):
# "detected_bounding_box": [31.1, 34.8, 0, 31.0, 34.9, 0]
# Insert (in "metadata"):
# "detected_bounding_box": [540-140, 960-140, 0, 540+140, 960+140, 0]

# Installing the package:
# pip install dist/wildfire_detector-0.1.4-py3-none-any.whl

import numpy as np
from wildfire_detector.function_class_TensorRT import ScanManager

# === Init ScanManager ===
sm = ScanManager()

# === Create dummy IR image ===
ir_height, ir_width = sm.config['image']['ir_size']
frame_ir = np.random.randint(0, 255, (ir_height, ir_width), dtype=np.uint8)

# === Create dummy metadata ===
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
        "field_of_view_deg": 2.2,
        "resolution_px": [1920, 1080],
    },
    "geolocation": {
        "transformation_matrix": np.eye(4, dtype=float).ravel(order="C").tolist(),
        "latitude": 31.0461, # NonUsed
        "longitude": 34.8516, # NonUsed
    },
    "investigation_parameters": {
        "detection_latitude": 31.0421,
        "detection_longitude": 34.8516,
        "detection_altitude": 0.0000,
        "detected_bounding_box": [31.1, 34.8, 0.0, 31.0, 34.9, 0.0]
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
print("\nStarting Phase 2 :")
rgb_height, rgb_width = sm.config['image']['rgb_size']
frame_rgb = np.random.randint(0, 255, (ir_height, ir_width, 3), dtype=np.uint8)
result_phase2 = sm.phase2(frame_rgb, metadata)
print("\nPhase 2 result:")
print(result_phase2)

