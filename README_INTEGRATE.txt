import numpy as np
from wildfire_detector.function_class_demo import ScanManager

# === Init ScanManager ===
sm = ScanManager()

# === Create dummy IR image ===
ir_height, ir_width = sm.config['image']['ir_size']
frame = np.random.randint(0, 255, (ir_height, ir_width), dtype=np.uint8)

# === Create dummy metadata ===
metadata = {
    "scan_parameters": {
        "current_scanned_frame_id": 3
    },
    "uav": {
        "altitude_agl_meters": 50.0
    },
    "payload": {
        "pitch_deg": 10.0,
        "field_of_view_deg": 45.0
    },
    "transformation_matrix": np.eye(4).tolist()  # Identity for test
}

# === Phase 0: Save reference frame and corner projection ===
sm.phase0(frame, metadata)

# === Phase 1: IR fire detection ===
results_phase1 = sm.phase1(frame, metadata)
print("\nPhase 1 results:")
for res in results_phase1:
    print(res)

# === Phase 2: RGB fire confirmation ===
# Use a dummy bbox (normally comes from phase1)
dummy_bbox = [32.1, 34.8, 32.0, 34.9]  # [top_lat, top_lon, bottom_lat, bottom_lon]
result_phase2 = sm.phase2(frame, dummy_bbox, metadata)
print("\nPhase 2 result:")
print(result_phase2)

