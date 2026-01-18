# Create the package:
# pip install build
# python -m build
# pip install dist/wildfire_detector-0.1.2-py3-none-any.whl

# Install the package - windows:
# pip install dist/wildfire_detector-0.1.1-py3-none-any.whl
# from wildfire_detector.function_class_demo import ScanManager

# Install/Uninstall the package - Linux:
# At the Terminal:
# cd /path/to/Flame2 (/home/malat/PycharmProject/Flame2)
# source .venv/bin/activate
# pip uninstall wildfire_detector
# pip install dist/wildfire_detector-0.1.1-py3-none-any.whl
# To run a .py file in the Terminal
# python Try.py

# Creating onnx file from the .pt file
# run TensorRT_Conversion.py file

# RT: - at the Jetson, after we got onnx file. - Just a backup - already runs inside function_classRT code
/usr/src/tensorrt/bin/trtexec \
  --onnx=wildfire_detector/resnet_fire_classifier.onnx \
  --saveEngine=wildfire_detector/resnet_fire_classifier_fp16.trt \
  --fp16 \
  --minShapes=input:1x3x254x254 \ #?
  --optShapes=input:3x3x254x254 \ #?
  --maxShapes=input:16x3x254x254 \ #?
  --shapes=input:1x3x254x254 #?

# In order to use RT class replace :
# from wildfire_detector.function_class_demo import ScanManager
# with:
# from wildfire_detector.function_class_demo_TensorRT import ScanManager

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
    "transformation_matrix": np.eye(4, dtype=float).ravel(order="C").tolist()  # Identity for test
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

