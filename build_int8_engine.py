"""
build_int8_engine.py
--------------------
Builds a TensorRT INT8 engine from an ONNX model using a calibration dataset.

This script uses TensorRT's Python API with an IInt8EntropyCalibrator2 to:
  1. Load calibration images from a folder (resized to model input size, normalized)
  2. Compute per-layer quantization scales (INT8 range mapping)
  3. Build and save the INT8 TensorRT engine

The calibration images should be representative samples from the training data
(~500-1000 images, mix of Fire and NoFire).

Requirements:
  - NVIDIA Jetson with TensorRT installed (tensorrt Python package)
  - ONNX model file (best_model.onnx)
  - PyCUDA (for GPU memory during calibration)

Usage:
  python build_int8_engine.py --onnx <path_to_onnx> --calib_dir <path_to_images> --output <output.trt>

Examples:
  python build_int8_engine.py \
    --onnx experiments/resnet18/checkpoints/best_model.onnx \
    --calib_dir Seperated_Dataset/FLAME_2/Fire \
    --output wildfire_detector/best_model_int8.trt

  python build_int8_engine.py \
    --onnx wildfire_detector/best_model.onnx \
    --calib_dir Seperated_Dataset/D_Fire/Fire/images \
    --output wildfire_detector/best_model_int8.trt \
    --num_calib 500
"""

import os
import sys
import argparse
import numpy as np
from glob import glob

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 — initializes CUDA context
except ImportError as e:
    print(f"ERROR: {e}")
    print("This script must run on the Jetson with TensorRT and PyCUDA installed.")
    sys.exit(1)


TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# ImageNet normalization (same as inference pipeline)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_and_preprocess_image(image_path, net_size):
    """Load an image, resize to net_size x net_size, normalize (ImageNet stats).
    Returns a float32 CHW array ready for the model."""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (net_size, net_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return img


def collect_calibration_images(calib_dir, net_size, num_calib=500):
    """Collect and preprocess calibration images from a directory."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(calib_dir, "**", ext), recursive=True))

    if len(image_paths) == 0:
        print(f"ERROR: No images found in {calib_dir}")
        sys.exit(1)

    # Shuffle and limit
    np.random.seed(42)
    np.random.shuffle(image_paths)
    image_paths = image_paths[:num_calib]

    print(f"Loading {len(image_paths)} calibration images from: {calib_dir}")
    images = []
    for path in image_paths:
        img = load_and_preprocess_image(path, net_size)
        if img is not None:
            images.append(img)

    print(f"Successfully loaded {len(images)} images (shape: 1x3x{net_size}x{net_size})")
    return images


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator that feeds preprocessed images to TensorRT for quantization."""

    def __init__(self, images, batch_size=8, cache_file="calibration.cache"):
        super().__init__()
        self.images = images
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.current_index = 0

        # Allocate GPU memory for one batch
        sample = images[0]
        self.batch_nbytes = sample.nbytes * batch_size
        self.d_input = cuda.mem_alloc(self.batch_nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.images):
            return None

        end_index = min(self.current_index + self.batch_size, len(self.images))
        batch = self.images[self.current_index:end_index]
        self.current_index = end_index

        # Pad last batch if needed
        if len(batch) < self.batch_size:
            pad = [np.zeros_like(batch[0])] * (self.batch_size - len(batch))
            batch = batch + pad

        batch_array = np.ascontiguousarray(np.stack(batch))
        cuda.memcpy_htod(self.d_input, batch_array)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[Calibrator] Reading cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"[Calibrator] Writing cache: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_int8_engine(onnx_path, calibrator, net_size, max_batch=16):
    """Build a TensorRT INT8 engine from ONNX model with calibration."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"[TRT] Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            sys.exit(1)

    # Builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # Allow FP16 fallback for unsupported layers
    config.int8_calibrator = calibrator

    # Dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape("input",
                      min=(1, 3, net_size, net_size),
                      opt=(3, 3, net_size, net_size),
                      max=(max_batch, 3, net_size, net_size))
    config.add_optimization_profile(profile)

    # Build
    print("[TRT] Building INT8 engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Engine build failed!")
        sys.exit(1)

    return serialized_engine


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT INT8 engine with calibration")
    parser.add_argument("--onnx", type=str, required=True,
                        help="Path to the ONNX model file")
    parser.add_argument("--calib_dir", type=str, required=True,
                        help="Path to folder with calibration images (Fire + NoFire recommended)")
    parser.add_argument("--output", type=str, default="wildfire_detector/best_model_int8.trt",
                        help="Output path for the INT8 engine file")
    parser.add_argument("--net_size", type=int, default=224,
                        help="Model input size (default: 224)")
    parser.add_argument("--num_calib", type=int, default=500,
                        help="Number of calibration images to use (default: 500)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Calibration batch size (default: 8)")
    parser.add_argument("--cache_file", type=str, default="calibration.cache",
                        help="Path to save/load calibration cache")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"ERROR: ONNX file not found: {args.onnx}")
        sys.exit(1)

    # Collect and preprocess calibration images
    images = collect_calibration_images(args.calib_dir, args.net_size, args.num_calib)

    # Create calibrator
    calibrator = Int8Calibrator(images, batch_size=args.batch_size, cache_file=args.cache_file)

    # Build engine
    serialized_engine = build_int8_engine(args.onnx, calibrator, args.net_size)

    # Save engine
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(serialized_engine)

    print(f"\nINT8 engine saved to: {args.output}")
    print(f"Calibration cache: {args.cache_file}")
    print(f"\nTo use: set trt_precision: \"int8\" in wildfire_detector/config.yaml")


if __name__ == "__main__":
    main()
