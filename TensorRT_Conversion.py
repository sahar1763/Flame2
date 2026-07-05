"""
TensorRT_Conversion.py
----------------------
Exports a trained PyTorch classification model to ONNX format.
The ONNX file is then used on the Jetson Orin Nano where:
  - FP16 engine is auto-built by ScanManager on first launch (via trtexec)
  - INT8 engine is built explicitly using build_int8_engine.py

Input:  An experiment folder produced by train.py
Output: best_model.onnx saved inside the experiment's checkpoints/ directory
"""

import torch
from torchvision import models
import yaml
import os
import argparse
import importlib.resources as pkg_resources

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
parser.add_argument("--experiment", type=str, required=True,
                    help="Path to experiment folder (contains checkpoints/ and config.yaml)")
args = parser.parse_args()

# --- Load training config from the experiment folder ---
config_path = os.path.join(args.experiment, "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# --- Locate the trained checkpoint ---
checkpoint_dir = os.path.join(args.experiment, "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
onnx_path = os.path.join(checkpoint_dir, "best_model.onnx")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

num_classes = 2

# --- Rebuild the model architecture (must match what train.py created) ---
model_name = config["training"]["model_name"]
model = getattr(models, model_name)(weights=None)

if "resnet" in model_name.lower():
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
elif "vgg" in model_name.lower() or "alexnet" in model_name.lower():
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(num_ftrs, num_classes)
elif "mobilenet" in model_name.lower():
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
else:
    raise ValueError(f"Unsupported model: {model_name}")

# --- Load trained weights into the model ---
model.load_state_dict(checkpoint["model_state"])
model.eval()

net_image_size = config["dataset"]["image_size"]

# --- Export to ONNX (dynamic batch axis for flexible inference) ---
dummy = torch.randn(1, 3, net_image_size, net_image_size)
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=13,
)

print(f"Exported ONNX : {onnx_path}")

# =============================================================================
# Usage:
#   python TensorRT_Conversion.py --experiment <path_to_experiment_folder>
#
# The experiment folder is created by train.py and contains:
#   <experiment_folder>/
#       config.yaml              <- has training.model_name and dataset.image_size
#       checkpoints/
#           best_model.pt        <- trained weights (input)
#           best_model.onnx      <- exported ONNX model (output of this script)
#
# On the Orin:
#   - Copy best_model.onnx to wildfire_detector/
#   - FP16 engine auto-builds on first ScanManager launch
#   - INT8 engine: run build_int8_engine.py separately
#
# Examples:
#   python TensorRT_Conversion.py --experiment experiments/2026-02-25_10-22-23_resnet18
#   python TensorRT_Conversion.py --experiment experiments/2026-02-26_08-15-00_mobilenet_v2
#   python TensorRT_Conversion.py --experiment experiments/2026-02-27_14-30-10_vgg16
#
# Supported models: resnet18, resnet34, resnet50, alexnet, vgg16, mobilenet_v2
# =============================================================================
