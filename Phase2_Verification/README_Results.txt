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
CURRENT WORKFLOW - AUTOMATED ORCHESTRATORS (use these)
================================================================================
The whole Chapter 5 matrix is now driven by two orchestrator scripts. They
locate each model's weights under Trained_Models/ automatically, build the
TensorRT engines they need, run all combinations, and aggregate the CSVs. You
do NOT need to hand-swap weights or edit config.yaml per model anymore.

Run BOTH from the Phase2_Verification/ folder ON THE ORIN.

--------------------------------------------------------------------------------
A) run_classifier_benchmarks.py  -> Sections 5.2-5.5 (single-image classifier)
--------------------------------------------------------------------------------
Matrix: 6 models x 3 precisions (fp32/fp16/int8) = 18 runs.

  # FULL command used for the report:
  python run_classifier_benchmarks.py \
      --images_dir ../Resized_TestSet_Cropped_Biggest \
      --labels_csv ../Resized_TestSet_Cropped_Biggest/labels.csv \
      --test_datasets FireSmokeDataset FireSmokeNEWdataset \
      --calib_dir ../Resized_TestSet_Cropped_Biggest \
      --num_calib 1000 --rebuild_engines \
      --trained_models ../Trained_Models/experiments_2nd \
      --output_dir classifier_results

Combinations (all optional except the dataset target):
  --models      subset of {alexnet vgg16 resnet18 resnet34 resnet50 mobilenet_v2}  (default: all 6)
  --precisions  subset of {fp32 fp16 int8}                                         (default: all 3)
  dataset       EITHER --images_dir + --labels_csv (+ --test_datasets)  OR  --dataset <Fire/+NoFire folder>
  --calib_dir   REQUIRED when int8 is requested and the engine must be built (both classes, recursively globbed)
  --num_calib   INT8 calibration images (default 500; report used 1000)
  --rebuild_engines   force rebuild .trt   |   --skip_engine_build  assume they exist
  --dry_run     print commands without executing

Output: classifier_results/<model>_<backend>/summary.csv + combined_classifier_summary.csv

--------------------------------------------------------------------------------
B) run_crop_benchmarks.py  -> Sections 5.6-5.7 (full RGB crop pipeline)
--------------------------------------------------------------------------------
Matrix: 6 models x 3 backends x 3 datasets x 3 crop strategies = 162 runs.

  # FULL command used for the report:
  python run_crop_benchmarks.py \
      --dataset_root ../Sampled_Seperated_Dataset \
      --trained_models ../Trained_Models/experiments_2nd \
      --backends pytorch tensorrt int8 \
      --strategies raw crops1 crops3 \
      --fire_labels_subdir labels_biggest \
      --calib_dir ../Sampled_Seperated_Dataset \
      --num_calib 1000 \
      --output_dir crop_results \
      --trtexec /usr/src/tensorrt/bin/trtexec

Combinations (all optional except --dataset_root):
  --models      subset of the 6 architectures                         (default: all 6)
  --backends    subset of {pytorch(FP32) tensorrt(FP16) int8(INT8)}   (default: pytorch)
  --strategies  subset of {raw crops1 crops3}                         (default: all 3)
                  raw    = whole frame, no crop
                  crops1 = single sqrt(2) crop around the bbox
                  crops3 = three-scale crop + majority vote
  --datasets    subset of sub-folders under --dataset_root            (default: all)
  --fire_labels_subdir  Fire bbox label folder                        (default: labels_biggest)
  --calib_dir   REQUIRED when int8 is in --backends (both classes, recursively globbed)
  --num_calib   INT8 calibration images (default 500; report used 1000)
  --no_restore  leave last-staged model in the package  |  --dry_run  print only

Output: crop_results/<dataset>/<model>_<backend>_<strategy>/summary.csv + combined_crop_summary.csv

--------------------------------------------------------------------------------
C) Figures + tables (run on the PC from the CSVs produced above)
--------------------------------------------------------------------------------
  # classifier sections 5.2-5.5:
  python generate_figures.py --results_dir classifier_results \
      --output_dir figures_classifier --weights_dir ../Trained_Models/experiments_2nd

  # crop sections 5.6-5.7:
  python generate_figures.py --results_dir crop_results \
      --output_dir figures_crop --weights_dir ../Trained_Models/experiments_2nd

  # precision-agreement proof (FP32 vs FP16 vs INT8, per image):
  python analyze_agreement.py --results_dir classifier_results

  # qualitative errors (section 5.8):
  python visualize_predictions.py --dataset ../Sampled_Seperated_Dataset/FireSmokeDataset \
      --predictions crop_results/FireSmokeDataset/resnet18_tensorrt_crops3/predictions.csv --errors

--weights_dir lets generate_figures.py read REAL params + weights_mb + onnx/fp16/int8
engine sizes straight from the model files (nothing hardcoded).


================================================================================
LEGACY SINGLE-MODEL WORKFLOW (below) - superseded by the orchestrators above
================================================================================


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
POST-PROCESSING TOOLS (PC-only, NO Orin re-run)
================================================================================
These operate purely on the predictions.csv files already produced by the Orin
benchmarks. They never call the model on the Orin again; every number is a
re-count of stored per-image predictions. All of them key on the PAIR
(image, true_label) because the IAI Fire/ and NoFire/ folders use INDEPENDENT
numbering, so the same filename (e.g. img_000001.jpg) exists in both folders and
must be disambiguated by its label.

--------------------------------------------------------------------------------
build_consensus_mistakes.py  — images that EVERY run got wrong
--------------------------------------------------------------------------------
Intersects the errors of many runs and lists only the (image, true_label) pairs
that were misclassified by ALL of them. Use it to find the "unresolvable" cases
independent of model/backend/strategy.

  --results_dir  results tree to scan (default crop_results/IAI_Datasets)
  --strategy     all | crops1 | crops3 | raw
                   all  -> intersect across ALL strategies+backends+archs
                           (IAI = 6 arch x 3 backend x 3 strategy = 54 runs)
                   cropsN/raw -> restrict to one strategy (18 runs each)
  --backend      all | pytorch | tensorrt | int8

  Output: <results_dir>/IAI_consensus_mistakes_<strategy>.csv
          columns: image, folder, true_label, error_type (FP/FN)

  Example (the 303-image "no model detected it" list used for filtering):
    python build_consensus_mistakes.py --strategy all

--------------------------------------------------------------------------------
recompute_filtered_metrics.py  — metrics with a set of images removed
--------------------------------------------------------------------------------
Recomputes precision/recall/F1/FPR/accuracy for every run BEFORE and AFTER
dropping an exclusion list, and reports the deltas. Read-only; writes one
comparison table.

  --results_dir  results tree to scan (default crop_results/IAI_Datasets)
  --exclude      CSV with columns image,true_label to drop
                   (default IAI_consensus_mistakes_all.csv)
  --strategy     all | crops1 | crops3 | raw
  --backend      all | pytorch | tensorrt | int8
  --out          optional output path

  Output: <results_dir>/IAI_metrics_filtered_<strategy>.csv
          before/after columns for recall, f1, precision, fpr, accuracy + counts

  Example:
    python recompute_filtered_metrics.py --strategy all

--------------------------------------------------------------------------------
materialize_filtered_iai.py  — write a full filtered results tree
--------------------------------------------------------------------------------
Creates a PARALLEL results directory with the excluded images physically removed
from each predictions.csv and each summary.csv recomputed (metric columns
updated, ALL timing columns copied unchanged). Originals are never modified.
The output tree has the exact same layout/schema as the input, so every
downstream tool (generate_figures.py, build_consensus_mistakes.py,
recompute_filtered_metrics.py) can point at it directly.

  --src      source results tree (default crop_results/IAI_Datasets)
  --dst      output tree         (default crop_results/IAI_Datasets_filtered)
  --exclude  CSV with columns image,true_label to drop
               (default IAI_consensus_mistakes_all.csv)

  Output: <dst>/<model>_<backend>_<strategy>/{predictions.csv, summary.csv}
          <dst>/combined_crop_summary.csv

  Example (build the filtered IAI used from now on):
    python materialize_filtered_iai.py
    python generate_figures.py --results_dir crop_results\IAI_Datasets_filtered --output_dir figures_iai_filtered --weights_dir ..\Trained_Models\experiments_2nd

--------------------------------------------------------------------------------
visualize_predictions.py  — NEW options (see file docstring for full list)
--------------------------------------------------------------------------------
  --fire_labels_subdir  Fire bbox label subfolder (default labels_biggest).
                        MUST match run_crop_benchmarks.py, else fire crops fall
                        back to a virtual center box (NOT benchmark-accurate).
  --weights             Load the EXACT benchmarked checkpoint so overlay scores
                        match the predictions.csv (otherwise the packaged
                        wildfire_detector/best_model.pt is used).
  Image lookup now uses true_label (Fire/NoFire share filenames). The "Final"
  title is taken from the CSV verdict so it can never contradict the FP/FN
  folder. Pair a 3-crop plot with a *_crops3 CSV (and a 1-crop plot with a
  *_crops1 CSV) so the panels and the verdict use the same strategy.


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
[✓] Post-processing: consensus mistakes / filtered metrics / filtered tree

ALL CODE IS IMPLEMENTED. Remaining work is execution on the Jetson Orin Nano.
