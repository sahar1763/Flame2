import torch
from torchvision import models
import yaml
import os
import importlib.resources as pkg_resources

config_path = r"wildfire_detector\config.yaml"

# Load config.yaml from package
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# ==== Load your PyTorch checkpoint ====
checkpoint_dir = r"experiments\2026-02-25_10-22-23_resnet18\checkpoints"
checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
onnx_path = os.path.join(checkpoint_dir, "best_model.onnx")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

num_classes = 2

# ---------- Initialize Model ----------
model_name = config["phase2"]["model_name"]
model = getattr(models, model_name)(weights=None)

if "resnet" in model_name.lower():
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
elif "vgg" in model_name.lower() or "alexnet" in model_name.lower():
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(num_ftrs, num_classes)
elif "densenet" in model_name.lower():
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, num_classes)

model.load_state_dict(checkpoint["model_state"])
model.eval()

net_image_size = config["phase2"]["net_image_size"]

# ==== Export to ONNX ====
dummy = torch.randn(1, 3, net_image_size, net_image_size) # TODO: Check and validate image size
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
