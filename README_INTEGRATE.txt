# For running the demo package:
# Instead of:
# from wildfire_detector.function_class_TensorRT import ScanManager
# Insert:
# from wildfire_detector.function_class_demo_TensorRT import ScanManager
# And instead of (in "metadata"):
# "detected_bounding_box": [31.1, 34.8, 31.0, 34.9]
# Insert (in "metadata"):
# "detected_bounding_box": [960-140, 540-140, 960+140, 540+140]

# Installing the package:
# pip install dist/wildfire_detector-0.1.3-py3-none-any.whl

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
        "roll_deg": 0.5,
        "pitch_deg": -1.2,
        "yaw_deg": 45.0,
    },
    "payload": {
        "pitch_deg": -12.0,
        "azimuth_deg": 128.0,
        "field_of_view_deg": 2.5,
        "resolution_px": [1920, 1080],
    },
    "geolocation": {
        "latitude": 31.0461,
        "transformation_matrix": np.eye(4).tolist(),
        "longitude": 34.8516,
    },
    "investigation_parameters": {
        "detection_latitude": 31.0421,
        "detection_longitude": 34.8516,
        "detected_bounding_box": [31.1, 34.8, 31.0, 34.9]
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
rgb_height, rgb_width = sm.config['image']['rgb_size']
frame_rgb = np.random.randint(0, 255, (ir_height, ir_width, 3), dtype=np.uint8)
result_phase2 = sm.phase2(frame_rgb, metadata)
print("\nPhase 2 result:")
print(result_phase2)

