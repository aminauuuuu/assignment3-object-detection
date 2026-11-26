"""
Assignment 3 - Part 1, step 1: export pretrained YOLO26 models to ONNX.

Run this ONLY in the separate "export" environment
(the one where you did: pip install -r requirements-export.txt).

What it does, for each model (yolo26n and yolo26s):
  1. downloads the pretrained .pt file (first time only)
  2. counts its parameters (the "knobs" inside the network)
  3. converts it to .onnx with exactly the tutorial's settings
  4. saves the class names to a .json file so OpenCV can show "cup" instead of "41"

Usage:
    python export_models.py                    # exports yolo26n and yolo26s
    python export_models.py best.pt            # later, in Part 2: export your fine-tuned model
"""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

names_to_export = sys.argv[1:] or ["yolo26n.pt", "yolo26s.pt"]

for name in names_to_export:
    model = YOLO(name)

    # Number of parameters = how many numbers the network learned.
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"{name}: {n_params:,} parameters")

    # Class names, e.g. {0: 'person', 1: 'bicycle', ...}  ->  saved as a list
    class_names = [model.names[i] for i in range(len(model.names))]

    # Export with the tutorial's settings (do not change these).
    onnx_path = model.export(format="onnx", imgsz=640, batch=1, dynamic=False,
                             half=False, opset=17, simplify=False,
                             end2end=True, nms=False, device="cpu")

    stem = Path(onnx_path).with_suffix("")
    with open(f"{stem}_classes.json", "w") as f:
        json.dump(class_names, f)
    with open(f"{stem}_info.json", "w") as f:
        json.dump({"model": name, "parameters": n_params,
                   "num_classes": len(class_names)}, f, indent=2)

    print(f"  -> {onnx_path}")
    print(f"  -> {stem}_classes.json, {stem}_info.json\n")
