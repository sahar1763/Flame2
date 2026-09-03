================================================================================
README_PHASE2_ORIN.txt
Phase 2 (RGB fire verification) — Benchmarking on the Jetson Orin Nano
================================================================================

PURPOSE
-------
End-to-end guide to reproduce the Chapter 5 results on the Jetson Orin Nano:
  - upload the trained models + test data to the Orin,
  - build the TensorRT FP16 and INT8 engines ON the Orin,
  - run the benchmark scripts,
  - generate the report figures/tables.

There are TWO separate measurement paths — do not confuse them:

  A) classifier_latency_benchmark.py  (Sections 5.3 and 5.5)
       Plain single-image classifier: whole image -> 224 -> ONE forward pass.
       NO bbox, NO crops, NO voting. Accuracy matches the training / W&B numbers.
       Used for: 6-model FP16 latency (5.3) and ResNet18 FP32/FP16/INT8 (5.5).

  B) benchmark_inference.py           (Section 5.6 only)
       Full ScanManager pipeline: multi-scale crops + majority vote, per-stage
       timing breakdown. Used ONLY for the pipeline-breakdown table.

Accuracy is reported from ONE source (the server test.py plain classifier). The
Orin runs are for LATENCY. FP32 predictions are numerically identical across
hardware, so this stays consistent.

IMPORTANT — two separate result folders (do NOT mix them):
  benchmark_results/           <- classifier_latency_benchmark.py  (5.3 / 5.5)
  benchmark_results_pipeline/  <- benchmark_inference.py           (5.6)
Both scripts name subfolders <model>_<backend>, so writing them to the same
folder would OVERWRITE each other. Keep them apart as shown throughout.


================================================================================
QUICK-START (AUTOMATED) — all 6 models in one command (recommended)
================================================================================
run_classifier_benchmarks.py loops the 6 models x 3 precisions (18 runs) for
you: it locates each model's experiment folder under --trained_models, builds
the FP16 engine (trtexec) and the INT8 engine (build_int8_engine.py, with a
FRESH per-model calibration cache), then runs classifier_latency_benchmark.py
for fp32/fp16/int8 and aggregates combined_classifier_summary.csv. No manual
per-model staging or "rm -f calibration.cache" needed.

Run FROM Phase2_Verification/ on the Orin (this is the exact command used):
    python run_classifier_benchmarks.py \
        --images_dir ../Resized_TestSet_Cropped_Biggest \
        --labels_csv ../Resized_TestSet_Cropped_Biggest/labels.csv \
        --test_datasets FireSmokeDataset FireSmokeNEWdataset \
        --calib_dir ../Resized_TestSet_Cropped_Biggest \
        --num_calib 1000 \
        --rebuild_engines \
        --output_dir classifier_results \
        --trained_models ../Trained_Models/experiments_2nd

  Notes:
    - --trained_models  : folder holding the <timestamp>_<model> experiment
                          folders (e.g. experiments_2nd/2026-08-11_17-15-06_alexnet).
    - --rebuild_engines : force fresh FP16/INT8 engines on the Orin's TensorRT
                          (do NOT reuse .trt files copied from another machine).
    - --num_calib 1000  : use all calibration images (default is 500).
    - INT8 cache is per-model and deleted before each build (no cross-model reuse).
    - Add --dry_run first to print every command without building anything.
    - Restrict with --models resnet18 (single model) or --precisions fp32 fp16.
  Output: classifier_results/<model>_<backend>/{predictions,summary}.csv
          + classifier_results/combined_classifier_summary.csv

Then generate the figures (can run on the PC from the CSVs, section 7):
    python Phase2_Verification/generate_figures.py --results_dir classifier_results --output_dir figures


================================================================================
QUICK-START (MANUAL) — one model, copy/paste (example: resnet18)
================================================================================
Use this only if you want to drive a single model by hand. For each new model:
replace resnet18 with the model name, point --weights at that model's files,
adjust config.yaml (section 6.1), then paste these blocks.

--- Build engines on the Orin (section 3) ---
    python build_fp16_engine.py --onnx wildfire_detector/best_model.onnx \
        --output wildfire_detector/best_model_fp16.trt
    rm -f calibration.cache
    python build_int8_engine.py --onnx wildfire_detector/best_model.onnx \
        --calib_dir calib_INT8 --output wildfire_detector/best_model_int8.trt --num_calib 1000

--- 5.5 classifier latency, 3 precisions (writes benchmark_results/) ---
    python Phase2_Verification/classifier_latency_benchmark.py --model_name resnet18 --precision fp32 \
        --weights wildfire_detector/best_model.pt \
        --images_dir TestSet --labels_csv TestSet/labels.csv \
        --test_datasets FireSmokeDataset FireSmokeNEWdataset --output_dir benchmark_results
    python Phase2_Verification/classifier_latency_benchmark.py --model_name resnet18 --precision fp16 \
        --weights wildfire_detector/best_model_fp16.trt \
        --images_dir TestSet --labels_csv TestSet/labels.csv \
        --test_datasets FireSmokeDataset FireSmokeNEWdataset --output_dir benchmark_results
    python Phase2_Verification/classifier_latency_benchmark.py --model_name resnet18 --precision int8 \
        --weights wildfire_detector/best_model_int8.trt \
        --images_dir TestSet --labels_csv TestSet/labels.csv \
        --test_datasets FireSmokeDataset FireSmokeNEWdataset --output_dir benchmark_results

--- 5.6 pipeline breakdown (writes benchmark_results_pipeline/) ---
    # set config.yaml first (section 6.1), then per precision:
    python Phase2_Verification/benchmark_inference.py --dataset <Fire_NoFire_subset> \
        --backend pytorch  --output_dir benchmark_results_pipeline
    python Phase2_Verification/benchmark_inference.py --dataset <Fire_NoFire_subset> \
        --backend tensorrt --output_dir benchmark_results_pipeline
    python Phase2_Verification/benchmark_inference.py --dataset <Fire_NoFire_subset> \
        --backend int8     --output_dir benchmark_results_pipeline

--- Figures (can run on the PC from the CSVs; section 7) ---
    python Phase2_Verification/generate_figures.py --results_dir benchmark_results --output_dir figures
    python Phase2_Verification/generate_figures.py --results_dir benchmark_results_pipeline --output_dir figures_pipeline


================================================================================
0. HARDWARE / SOFTWARE PREREQUISITES (Orin)
================================================================================
Confirm these are importable on the Orin (JetPack provides most):
    python3 -c "import tensorrt, torch, torchvision, cv2, numpy, pandas, yaml, tqdm; print('ok')"

pycuda is required for the TensorRT inference path. If missing, see the
"pycuda install" section at the BOTTOM of this file.

---------------------------------------------------------------------------
INSTALL THE wildfire_detector PACKAGE AS EDITABLE  (do this ONCE, important!)
---------------------------------------------------------------------------
benchmark_inference.py (5.6) drives the ScanManager class from the INSTALLED
wildfire_detector package. ScanManager resolves config.yaml AND the .trt engine
files from wherever the package is installed. If a stale wheel is installed, it
looks in site-packages/ (not your repo) and you get errors like:
    - TypeError: __init__() got an unexpected keyword argument 'verbose'
    - config.yaml not found ... site-packages/wildfire_detector/
    - INT8 engine not found ... site-packages/wildfire_detector/best_model_int8.trt

Fix (run once, from the FireDrone repo root):
    pip uninstall wildfire_detector -y
    pip uninstall wildfire_detector -y      # run twice; sometimes two copies exist
    rm -rf .venv/lib/python3.10/site-packages/wildfire_detector   # remove leftovers
    pip install -e .

Verify it now points at your REPO folder (not site-packages):
    python -c "import wildfire_detector, os; print(os.path.dirname(wildfire_detector.__file__))"
    # must print .../FireDrone/wildfire_detector

After this, the engines you build in wildfire_detector/ are found automatically,
and you NEVER rebuild the wheel again -- source edits + swapped model files apply
live. (classifier_latency_benchmark.py does NOT use the package, so it is
unaffected either way -- it loads weights straight from --weights.)

CUDA must be on PATH before building engines / installing pycuda:
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    nvcc --version        # must print a version

(Optional) put the Orin in max-performance mode for stable latency numbers:
    sudo nvpmodel -m 0
    sudo jetson_clocks


================================================================================
1. WHAT TO PREPARE ON THE SERVER / PC (before uploading)
================================================================================
Engines are hardware-specific and MUST be built on the Orin, but the ONNX and
the calibration set are produced on the server/PC.

1.1  Export ONNX from each trained checkpoint (server):
        python TensorRT_Conversion.py --experiment experiments/<run>
        # -> experiments/<run>/checkpoints/best_model.onnx

     Do this for every model you benchmark (6 models for 5.3):
        resnet18, resnet34, resnet50, alexnet, vgg16, mobilenet_v2
     Give each ONNX/PT a distinct name or keep per-model subfolders so they
     don't overwrite each other.

1.2  Build the INT8 calibration set (PC) — TRAINING pool only, never test:
        python dataset_handling/Build_INT8_Calibration_Set.py --num 500 --clean
        # --num is PER CLASS -> 500 Fire + 500 NoFire = 1000 images in calib_INT8/

1.3  Prepare the test set (the SAME set the models were tested on):
     TestSet layout = flat images + labels.csv (columns: id, dataset, fire).
     The test datasets are the held-out FireSmoke sets:
        FireSmokeDataset, FireSmokeNEWdataset


================================================================================
2. WHAT TO UPLOAD TO THE ORIN
================================================================================
Recommended folder layout on the Orin (mirrors the repo):

    FireDrone/
    ├── wildfire_detector/
    │   ├── best_model.onnx          <- per model (input to engine builders)
    │   ├── best_model.pt            <- per model (needed for FP32 run only)
    │   ├── config.yaml
    │   └── TensorRT_infer.py        (+ the rest of the package)
    ├── build_fp16_engine.py
    ├── build_int8_engine.py
    ├── Phase2_Verification/
    │   ├── classifier_latency_benchmark.py
    │   ├── benchmark_inference.py
    │   └── generate_figures.py
    ├── calib_INT8/                  <- the 1000 calibration images (for INT8)
    └── TestSet/
        ├── <flat test images...>
        └── labels.csv

Minimum per model:
  - best_model.onnx  (required — engines are built from it)
  - best_model.pt    (only if you also want the FP32 baseline number)

NOTE: the .trt engine files are resolved INSIDE wildfire_detector/ by the full
pipeline (config keys), so build/keep them there.


================================================================================
2b. SWITCHING TO A NEW MODEL  (do this each time before section 3)
================================================================================
Because ScanManager reads from wildfire_detector/ (editable install, section 0),
switching architectures = swap the files in that folder + set model_name. No
reinstall, no rebuild.

    # 1) put the new model's exports into the package folder
    cp <new>/best_model.onnx  wildfire_detector/best_model.onnx
    cp <new>/best_model.pt    wildfire_detector/best_model.pt

    # 2) set the architecture in wildfire_detector/config.yaml
    #    model_name: \"resnet34\"       (options: resnet18/34/50, alexnet, vgg16, mobilenet_v2)

    # 3) build engines (section 3) and run benchmarks (sections 4/5/6)

Per-model checklist (the 3 easy-to-forget items):
  [ ] rm -f calibration.cache  BEFORE the INT8 build (see 3.2)
  [ ] config.yaml model_name matches the new architecture   (for 5.6)
  [ ] --model_name in the classifier commands matches too   (for 5.3/5.5)


================================================================================
3. BUILD THE ENGINES (on the Orin)
================================================================================
Do this per model. Swap the --onnx / --output paths for each architecture.

3.1  FP16 engine (no calibration needed):
        python build_fp16_engine.py \
            --onnx wildfire_detector/best_model.onnx \
            --output wildfire_detector/best_model_fp16.trt

     This is functionally identical to the trtexec auto-build inside ScanManager
     (same FP16 flag, same dynamic profile min=1 / opt=3 / max=16).

3.2  INT8 engine (needs the calibration set from step 1.2):
        python build_int8_engine.py \
            --onnx wildfire_detector/best_model.onnx \
            --calib_dir calib_INT8 \
            --output wildfire_detector/best_model_int8.trt \
            --num_calib 1000
     # build_int8_engine.py --num_calib is the OVERALL total (not per class).

     ----------------------------------------------------------------------
     RESET THE CALIBRATION CACHE BETWEEN DIFFERENT MODELS  (critical!)
     ----------------------------------------------------------------------
     NOTE: run_classifier_benchmarks.py / run_crop_benchmarks.py already do this
     for you — each model gets its own per-model .calib_cache next to its engine
     and it is DELETED before every build (Option A + Option B combined). The
     steps below are only for driving build_int8_engine.py manually.

     How the cache works (build_int8_engine.py):
       - Default cache file:  calibration.cache  (written in the CURRENT dir).
       - Controlled by the --cache_file argument.
       - On build, read_calibration_cache() reuses the file IF IT EXISTS and
         SKIPS real calibration. The activation ranges are model-specific, so
         reusing another model's cache produces a WRONG / degraded INT8 engine.

     You have TWO safe options:

     Option A — delete the cache before each new architecture:
        rm -f calibration.cache
        python build_int8_engine.py \
            --onnx wildfire_detector/resnet34.onnx \
            --calib_dir calib_INT8 \
            --output wildfire_detector/resnet34_int8.trt \
            --num_calib 1000

     Option B — give each model its own cache file (no deletion needed):
        python build_int8_engine.py \
            --onnx wildfire_detector/resnet34.onnx \
            --calib_dir calib_INT8 \
            --output wildfire_detector/resnet34_int8.trt \
            --num_calib 1000 \
            --cache_file resnet34_calibration.cache

     Per-model loop example (Option A, bash):
        for m in resnet18 resnet34 resnet50 alexnet vgg16 mobilenet_v2; do
            rm -f calibration.cache
            python build_int8_engine.py \
                --onnx wildfire_detector/${m}.onnx \
                --calib_dir calib_INT8 \
                --output wildfire_detector/${m}_int8.trt \
                --num_calib 1000
        done

     NOTE: re-building the SAME model? Keep the cache to save time. It is ONLY
     when the architecture changes that the cache must be reset.


================================================================================
4. SECTION 5.3 — 6-model FP16 latency
================================================================================
Run once per model (FP16). Use labels.csv mode restricted to the test sets so
the images/labels match training exactly:

    python Phase2_Verification/classifier_latency_benchmark.py \
        --model_name resnet18 --precision fp16 \
        --weights wildfire_detector/best_model_fp16.trt \
        --images_dir TestSet --labels_csv TestSet/labels.csv \
        --test_datasets FireSmokeDataset FireSmokeNEWdataset \
        --output_dir benchmark_results

Repeat with --model_name in:
    resnet18, resnet34, resnet50, alexnet, vgg16, mobilenet_v2
(each with its own best_model_fp16.trt built from that model's ONNX).

--test_datasets values MUST match the 'dataset' column in labels.csv exactly.


================================================================================
5. SECTION 5.5 — ResNet18 FP32 vs FP16 vs INT8
================================================================================
Same script, one run per precision (weights differ per precision):

    # FP32 (PyTorch checkpoint)
    python Phase2_Verification/classifier_latency_benchmark.py \
        --model_name resnet18 --precision fp32 \
        --weights wildfire_detector/best_model.pt \
        --images_dir TestSet --labels_csv TestSet/labels.csv \
        --test_datasets FireSmokeDataset FireSmokeNEWdataset

    # FP16 (TensorRT engine)
    python Phase2_Verification/classifier_latency_benchmark.py \
        --model_name resnet18 --precision fp16 \
        --weights wildfire_detector/best_model_fp16.trt \
        --images_dir TestSet --labels_csv TestSet/labels.csv \
        --test_datasets FireSmokeDataset FireSmokeNEWdataset

    # INT8 (TensorRT engine)
    python Phase2_Verification/classifier_latency_benchmark.py \
        --model_name resnet18 --precision int8 \
        --weights wildfire_detector/best_model_int8.trt \
        --images_dir TestSet --labels_csv TestSet/labels.csv \
        --test_datasets FireSmokeDataset FireSmokeNEWdataset

Optional args: --net_size 224 (default), --warmup 20 (default, discarded).


================================================================================
6. SECTION 5.6 — full pipeline breakdown (crops + vote)
================================================================================
This uses the ScanManager pipeline via benchmark_inference.py and reads the
model + engine from wildfire_detector/config.yaml.

6.1  Set the config keys in wildfire_detector/config.yaml:
        model_name: "resnet18"            # arch of the loaded checkpoint/engine
        trt_engine_fp16: "best_model_fp16.trt"
        trt_engine_int8: "best_model_int8.trt"

     NOTE: you do NOT need to change trt_precision by hand. benchmark_inference.py
     writes a temporary patched config and selects the precision from --backend
     automatically (pytorch/tensorrt/int8). The ONLY per-model config edit is
     model_name. (This relies on the editable install from section 0 so that
     ScanManager honors the temp config_path.)

6.2  Run (Fire/NoFire folder structure required for this one), one run per
     precision. Only the model_inference stage changes with precision; all other
     stages (crop extraction, resize, voting) are identical.

     # FP32 (uses best_model.pt -- NO engine, cheapest on disk)
     #   config: model_name: "resnet18"   (precision auto-selected from --backend)
        python Phase2_Verification/benchmark_inference.py \
            --dataset <path_to_Fire_NoFire_subset> \
            --backend pytorch \
            --output_dir benchmark_results_pipeline

     # FP16 (uses best_model_fp16.trt)
        python Phase2_Verification/benchmark_inference.py \
            --dataset <path_to_Fire_NoFire_subset> \
            --backend tensorrt \
            --output_dir benchmark_results_pipeline

     # INT8 (uses best_model_int8.trt)
        python Phase2_Verification/benchmark_inference.py \
            --dataset <path_to_Fire_NoFire_subset> \
            --backend int8 \
            --output_dir benchmark_results_pipeline

     --backend: pytorch = FP32, tensorrt = FP16, int8 = INT8.

     TIP: build a small Fire/NoFire timing subset (accuracy does not matter here,
     only per-stage timing) with the helper script -- it preserves the
     Fire/images + Fire/labels + NoFire structure and copies matching labels:
        python dataset_handling/Sample_Subset.py --src <full_Fire_NoFire_dataset> \
            --dst TimingSubset --count 50
     The input image size is irrelevant: the pipeline pastes each image onto the
     config image.rgb_size canvas (e.g. 1080x1920), so timings reflect the real
     deployment resolution regardless of source image size.

Write these to benchmark_results_pipeline/ (NOT benchmark_results/) so they do
not overwrite the classifier runs -- both scripts use the same <model>_<backend>
subfolder naming.
Use a SMALL full-resolution subset here -- this path is about per-stage timing,
not accuracy.  IMPORTANT: these latency numbers are hardware-specific, so this
script MUST run on the Orin (do NOT run it on the PC).


================================================================================
7. GENERATE THE FIGURES / TABLES
================================================================================
Both benchmark scripts write the SAME summary.csv schema into per-run folders,
but into TWO separate result dirs, so generate figures from each one:

    # 5.3 / 5.5 classifier results
    python Phase2_Verification/generate_figures.py \
        --results_dir benchmark_results \
        --output_dir figures

    # 5.6 pipeline-breakdown results
    python Phase2_Verification/generate_figures.py \
        --results_dir benchmark_results_pipeline \
        --output_dir figures_pipeline

Produces: confusion matrices (Fig 9), inference bar chart (Fig 10),
accuracy/latency trade-off scatter (Fig 11), precision comparison (Fig 12),
pipeline breakdown (5.6) and a combined table.

RUN THIS ANYWHERE (e.g. your PC): generate_figures.py only reads the CSV files
(pandas), it does NOT load any model, engine, CUDA or images. To save space on
the Orin, copy just the small benchmark_results/ folder to the PC and run it
there. Multi-model charts (Fig 10/11) need >=2 models present; with a single
model you still get its confusion matrix, precision comparison and table. The
script re-reads whatever is in benchmark_results/, so results ACCUMULATE across
runs/sessions -- add more models later and re-run to fill in the cross-model
charts.


================================================================================
8. OUTPUT LAYOUT
================================================================================
    benchmark_results/            (classifier_latency_benchmark.py -- 5.3 / 5.5)
      <model>_<backend>/
        predictions.csv   image, true_label, pred_label, confidence, model_inference_ms
        summary.csv       model, backend, precision, accuracy, fire_precision/recall/f1,
                          fpr, fnr, TP, FP, FN, TN, mean/std/min/p95/max_model_inference_ms
    benchmark_results_pipeline/   (benchmark_inference.py -- 5.6)
      <model>_<backend>/
        predictions.csv + summary.csv  (with per-stage pipeline timings)
    figures/            <- from benchmark_results
    figures_pipeline/   <- from benchmark_results_pipeline

backend label: fp32 -> pytorch, fp16 -> tensorrt, int8 -> int8.


================================================================================
9. COMMON PITFALLS
================================================================================
  - --test_datasets names must match labels.csv 'dataset' column EXACTLY,
    otherwise the run silently gets 0 images.
  - Delete calibration.cache (or use a per-model --cache_file) between different
    architectures when building INT8, or the wrong ranges get reused. See 3.2.
  - ONNX input tensor must be named "input" for the ScanManager trtexec path
    (build_fp16_engine.py reads the name from the graph, so it is safe either way).
  - Build engines ON the Orin — never copy engines from the server/PC.
  - Keep --net_size consistent (224) everywhere.


================================================================================
10. pycuda INSTALL (Jetson) — "No module named pycuda"
================================================================================
1) Ensure CUDA is on PATH (see section 0), verify:  nvcc --version
   Persist it:
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        source ~/.bashrc

2) Install build deps + pycuda:
        sudo apt-get update
        sudo apt-get install -y python3-dev build-essential
        pip3 install pycuda
   If the build fails, pin a version or use apt:
        pip3 install pycuda==2024.1
        # or:
        sudo apt-get install -y python3-pycuda

3) Verify:
        python3 -c "import pycuda.autoinit; import pycuda.driver as drv; print('pycuda OK:', drv.Device(0).name())"

Note: inside a virtualenv, create it with --system-site-packages OR export the
CUDA env vars inside the venv so nvcc is found during the pycuda build.
================================================================================
