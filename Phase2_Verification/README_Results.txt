================================================================================
Chapter 5 - Results: Step-by-Step Execution Guide
================================================================================

All commands assume working directory: j:\ControlProjects\FireDrone\phase2_verification
Target hardware for final numbers: NVIDIA Jetson Orin Nano


================================================================================
PREREQUISITES
================================================================================
- Trained model weights for each architecture (best_model.pt) placed in wildfire_detector/
- TensorRT engines (best_model_fp16.trt, best_model_int8.trt) converted on the Jetson
- D_Fire dataset at: ..\Seperated_Dataset\D_Fire (Fire/images, Fire/labels, NoFire/images)
- wildfire_detector/config.yaml → phase2.model_name set to the current model being tested
- Python environment with: torch, torchvision, numpy, pandas, opencv-python, matplotlib, tqdm, pyyaml


================================================================================
Section 5.1 — Candidate Model Evaluation Setup
================================================================================
Description:
  No code to run. This section describes the evaluation protocol.
  Table 8 is written manually in the thesis.


================================================================================
Section 5.2 — Classification Results on the Held-Out Test Set (Table 9, Fig 9)
================================================================================
Description:
  Run benchmark on D_Fire for each of the 6 models using PyTorch FP32 backend.
  Produces predictions.csv + summary.csv with accuracy, precision, recall, F1, FPR, FNR.
  NOTE: Must swap model weights + config.yaml model_name for each architecture.

File: benchmark_inference.py

Steps (repeat for each model):
  1. Place trained weights as wildfire_detector/best_model.pt
  2. Edit wildfire_detector/config.yaml → phase2.model_name: "<model_name>"
  3. Run:

  python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend pytorch
  python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend pytorch
  python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend pytorch
  python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend pytorch
  python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend pytorch
  python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend pytorch

Models to test (set model_name in config.yaml for each):
  - alexnet
  - vgg16
  - resnet18
  - resnet34
  - resnet50
  - mobilenet_v2

Output:
  benchmark_results/<model_name>_pytorch/predictions.csv
  benchmark_results/<model_name>_pytorch/summary.csv

Status: IMPLEMENTED ✓ | NOT YET RUN on Jetson


================================================================================
Section 5.3 — Runtime and Computational Comparison (Table 10, Fig 10)
================================================================================
Description:
  Inference latency per model measured on the Jetson Orin Nano (FP32).
  Data comes from the SAME benchmark runs as Section 5.2 (mean_model_inference_ms column
  in summary.csv). No additional runs needed.

File: benchmark_inference.py (same runs as 5.2)
      generate_figures.py (for Figure 10)

Command (after all 5.2 runs complete):
  python generate_figures.py --results_dir benchmark_results --output_dir figures

Output:
  figures/fig10_inference_time_pytorch.png

Status: IMPLEMENTED ✓ | NOT YET RUN on Jetson


================================================================================
Section 5.4 — Accuracy-Latency Trade-Off and Final Model Selection (Table 11, Fig 11)
================================================================================
Description:
  Combined table and scatter plot of F1-score vs inference time.
  Data comes from the SAME benchmark runs as Section 5.2.
  generate_figures.py creates the scatter plot automatically.

File: generate_figures.py (same command as 5.3)

Command:
  python generate_figures.py --results_dir benchmark_results --output_dir figures

Output:
  figures/fig11_tradeoff_pytorch.png
  figures/combined_results_table.csv  (use for Table 11 in thesis)

Status: IMPLEMENTED ✓ | NOT YET RUN on Jetson


================================================================================
Section 5.5 — TensorRT Optimization Results (Table 12, Fig 12)
================================================================================
Description:
  Compare ResNet18 across FP32, FP16, INT8 on the Jetson.
  Requires TensorRT engines pre-converted on the Jetson (see TensorRT_Conversion.py).
  Run benchmark 2 more times for the selected model (resnet18) with tensorrt and int8.

File: benchmark_inference.py

Prerequisites:
  - wildfire_detector/best_model_fp16.trt  (FP16 engine, converted on Jetson)
  - wildfire_detector/best_model_int8.trt  (INT8 engine, converted on Jetson)
  - config.yaml → phase2.model_name: "resnet18"

Commands:
  python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend tensorrt
  python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend int8

Then generate figures:
  python generate_figures.py --results_dir benchmark_results --output_dir figures

Output:
  benchmark_results/resnet18_tensorrt/summary.csv
  benchmark_results/resnet18_int8/summary.csv
  figures/resnet18/precision_comparison.png  (Fig 12)

Status: IMPLEMENTED ✓ | NOT YET RUN on Jetson (need TRT engines)


================================================================================
Section 5.6 — Full RGB Verification Pipeline Runtime (Table 13)
================================================================================
Description:
  7-stage pipeline timing breakdown using TensorRT FP16 on the Jetson.
  Data comes from the Section 5.5 tensorrt benchmark run (mean/std/min/max per stage).
  Also produces per-model pipeline breakdown stacked bar chart.

File: benchmark_inference.py (data from 5.5 run)
      generate_figures.py (for pipeline breakdown figure)

Command:
  python generate_figures.py --results_dir benchmark_results --output_dir figures

Output:
  figures/resnet18/pipeline_breakdown.png
  figures/fig_pipeline_breakdown_tensorrt.png
  figures/combined_results_table.csv (contains all 7 stage columns for Table 13)

Timing stages measured (7 stages):
  1. input_and_metadata
  2. region_validation
  3. crop_extraction
  4. resize_and_preprocess
  5. batch_construction
  6. model_inference
  7. majority_vote

Status: IMPLEMENTED ✓ | NOT YET RUN on Jetson


================================================================================
Section 5.7 — Multi-Scale Crop Strategy Evaluation (Table 14, Fig 13)
================================================================================
Description:
  Compare single-crop (√2 only) vs full 3-crop majority vote.
  Uses --crop_factors CLI override to test different configurations.

File: benchmark_inference.py

Commands (all with resnet18 + FP16 on Jetson):
  # Single crop (middle scale only): √2 ≈ 1.4142
  python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend tensorrt --crop_factors "1.4142"

  # Two crops: √1.5, √4
  python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend tensorrt --crop_factors "1.2247,2.0"

  # Three crops (default): √1.5, √2, √4  (no --crop_factors needed)
  python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend tensorrt

Output:
  benchmark_results/resnet18_tensorrt_crops_1.4142/summary.csv
  benchmark_results/resnet18_tensorrt_crops_1.2247_2.0/summary.csv
  benchmark_results/resnet18_tensorrt/summary.csv  (already from 5.5)

  Compare fire_precision, fire_recall, fire_f1, fpr, fnr across configurations.

Figure 13 (crop visualization example):
  python visualize_predictions.py --dataset ..\Seperated_Dataset\D_Fire --predictions benchmark_results\resnet18_tensorrt\predictions.csv --images "WEB07375.jpg"

Status: IMPLEMENTED ✓ | NOT YET RUN on Jetson


================================================================================
Section 5.8 — Qualitative Error Analysis (Fig 14, Fig 15, Table 15)
================================================================================
Description:
  Visual inspection of false positives and false negatives.
  Generates crop overlay plots showing per-crop predictions for misclassified images.

File: visualize_predictions.py

Commands:
  # All errors (FP + FN)
  python visualize_predictions.py --dataset ..\Seperated_Dataset\D_Fire --predictions benchmark_results\resnet18_tensorrt\predictions.csv --errors

  # Only false positives (for Figure 14)
  python visualize_predictions.py --dataset ..\Seperated_Dataset\D_Fire --predictions benchmark_results\resnet18_tensorrt\predictions.csv --fp_only

  # Only false negatives (for Figure 15)
  python visualize_predictions.py --dataset ..\Seperated_Dataset\D_Fire --predictions benchmark_results\resnet18_tensorrt\predictions.csv --fn_only

  # Specific images for hand-picked examples
  python visualize_predictions.py --dataset ..\Seperated_Dataset\D_Fire --predictions benchmark_results\resnet18_tensorrt\predictions.csv --images "img1.jpg,img2.jpg"

Output:
  visualizations/false_positives/FP_<name>.png
  visualizations/false_negatives/FN_<name>.png
  visualizations/selected/OK_<name>.png

Status: IMPLEMENTED ✓ | NOT YET RUN (needs predictions.csv from a benchmark run first)


================================================================================
BONUS: compare_bbox_overlay.py
================================================================================
Description:
  Utility to visualize YOLO bounding boxes overlaid on D_Fire images.
  Useful for verifying label quality and debugging bbox issues.
  Not directly a thesis figure but helpful for understanding dataset.

File: compare_bbox_overlay.py

Command:
  python compare_bbox_overlay.py --image WEB07375.jpg --save
  python compare_bbox_overlay.py --image WEB07375.jpg --org_labels ..\Seperated_Dataset\D_Fire_Org\Fire\labels --save

Output:
  visualizations/bbox_overlay_<name>.png


================================================================================
FULL EXECUTION ORDER (on Jetson Orin Nano)
================================================================================

Step 1: Train all 6 models (training_framework/train.py — already done)
Step 2: Convert selected model to TensorRT (TensorRT_Conversion.py — on Jetson)

Step 3: Run benchmarks (for each model, swap weights + config):
  FOR model IN [alexnet, vgg16, resnet18, resnet34, resnet50, mobilenet_v2]:
    → set config.yaml model_name, place best_model.pt
    → python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend pytorch

Step 4: Run TensorRT benchmarks (resnet18 only):
  → python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend tensorrt
  → python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend int8

Step 5: Run multi-crop comparison (resnet18 + tensorrt):
  → python benchmark_inference.py --dataset ..\Seperated_Dataset\D_Fire --backend tensorrt --crop_factors "1.4142"

Step 6: Generate all figures:
  → python generate_figures.py --results_dir benchmark_results --output_dir figures

Step 7: Generate error visualizations:
  → python visualize_predictions.py --dataset ..\Seperated_Dataset\D_Fire --predictions benchmark_results\resnet18_tensorrt\predictions.csv --errors


================================================================================
FILES SUMMARY
================================================================================

benchmark_inference.py     — Main benchmark script (sections 5.2-5.7)
generate_figures.py        — Figure generation from results CSVs (Fig 9-12 + pipeline)
visualize_predictions.py   — Crop visualization for error analysis (section 5.8, Fig 13-15)
compare_bbox_overlay.py    — YOLO bbox overlay utility (debugging/exploration)

wildfire_detector/config.yaml           — Model config (model_name, crop_factors, etc.)
wildfire_detector/function_class_demo.py — PyTorch ScanManager (FP32 inference)
wildfire_detector/function_class_demo_TensorRT.py — TensorRT ScanManager (FP16/INT8)
wildfire_detector/utils_phase2_flow.py  — Core functions (crop, predict, plot)


================================================================================
IMPLEMENTATION STATUS
================================================================================

✓ = Code implemented and tested
✗ = Not yet implemented

[✓] Section 5.2 — Classification benchmark (all models × pytorch)
[✓] Section 5.3 — Runtime comparison (from same data)
[✓] Section 5.4 — Trade-off scatter plot
[✓] Section 5.5 — TensorRT FP16/INT8 comparison
[✓] Section 5.6 — 7-stage pipeline breakdown
[✓] Section 5.7 — Multi-crop strategy (--crop_factors override)
[✓] Section 5.8 — Error visualization (visualize_predictions.py)
[✓] Figure generation (generate_figures.py)
[✓] Bbox overlay utility (compare_bbox_overlay.py)

ALL CODE IS IMPLEMENTED. Remaining work is execution on the Jetson Orin Nano.
