import torch
from torchvision import models

# ==== Load your PyTorch checkpoint ====
checkpoint = torch.load("resnet_fire_classifier.pt", map_location="cpu", weights_only=True)

num_classes = 2
resnet = models.resnet18(weights=None)
num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, num_classes)
resnet.load_state_dict(checkpoint["model_state"])
resnet.eval()

# ==== Export to ONNX ====
dummy = torch.randn(1, 3, 254, 254) # TODO: Check and validate image size
torch.onnx.export(
    resnet,
    dummy,
    "resnet_fire_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=13,
)

print("Exported ONNX -> resnet_fire_classifier.onnx")
