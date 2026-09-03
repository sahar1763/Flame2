"""
build_fp16_engine.py
--------------------
Builds a TensorRT FP16 engine from an ONNX model on the Jetson Orin Nano.

This is the FP16 counterpart of build_int8_engine.py. Unlike INT8, FP16 needs
NO calibration data — it simply enables half-precision kernels while building.

The dynamic-shape optimization profile matches build_int8_engine.py
(min=1, opt=3, max=16) so the resulting engine works both for:
  - single-image inference (batch 1)  -> classifier_latency_benchmark.py  (5.3/5.5)
  - the 3-crop verification pipeline (batch 3) -> benchmark_inference.py    (5.6)

Requirements:
  - NVIDIA Jetson with TensorRT installed (tensorrt Python package)
  - ONNX model file (best_model.onnx)

Usage:
  python build_fp16_engine.py --onnx <path_to_onnx> --output <output.trt>

Examples:
  python build_fp16_engine.py \
    --onnx wildfire_detector/best_model.onnx \
    --output wildfire_detector/best_model_fp16.trt

  python build_fp16_engine.py \
    --onnx experiments/resnet18/checkpoints/best_model.onnx \
    --output wildfire_detector/best_model_fp16.trt \
    --net_size 224
"""

import os
import sys
import argparse

try:
    import tensorrt as trt
except ImportError as e:
    print(f"ERROR: {e}")
    print("This script must run on the Jetson with TensorRT installed.")
    sys.exit(1)


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_fp16_engine(onnx_path, net_size, max_batch=16):
    """Build a TensorRT FP16 engine from an ONNX model (no calibration needed)."""
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

    if not builder.platform_has_fast_fp16:
        print("WARNING: platform reports no fast FP16 support; building anyway.")

    # Builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
    config.set_flag(trt.BuilderFlag.FP16)

    # Dynamic shapes (same profile as build_int8_engine.py)
    input_name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name,
                      min=(1, 3, net_size, net_size),
                      opt=(3, 3, net_size, net_size),
                      max=(max_batch, 3, net_size, net_size))
    config.add_optimization_profile(profile)

    # Build
    print("[TRT] Building FP16 engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Engine build failed!")
        sys.exit(1)

    return serialized_engine


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT FP16 engine (no calibration)")
    parser.add_argument("--onnx", type=str, required=True,
                        help="Path to the ONNX model file")
    parser.add_argument("--output", type=str, default="wildfire_detector/best_model_fp16.trt",
                        help="Output path for the FP16 engine file")
    parser.add_argument("--net_size", type=int, default=224,
                        help="Model input size (default: 224)")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"ERROR: ONNX file not found: {args.onnx}")
        sys.exit(1)

    serialized_engine = build_fp16_engine(args.onnx, args.net_size)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(serialized_engine)

    print(f"\nFP16 engine saved to: {args.output}")
    print(f"\nTo use: set trt_precision: \"fp16\" in wildfire_detector/config.yaml")


if __name__ == "__main__":
    main()
