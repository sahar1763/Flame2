Flame2/                             ← Project root folder
│
├── wildfire_detector/              ← Python package
│   ├── _init_.py
│   ├── functions_class.py          ← Contains the ScanManager class
│   ├── functions_class_TensorRT.py ← Contains the ScanManager class
│   ├── TensorRT_infer.py           ← TRTInference class for TensorRT
│   ├── config.yaml                 ← Configuration file
│   ├── utils_Frame.py              ← Utils for Phase1
│   ├── utils_phase2_flow.yaml      ← Utils for Phase2
│   ├── best_model.onnx             ← Model Weights file for Orin
│   └── best_model.pt               ← Model Weights file
│
├── setup.py                        ← Package installation script
├── README.md                       ← Package description
├── requirements.txt                ← Dependency list