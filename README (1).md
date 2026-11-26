# Assignment 3: Object detection with YOLO26 in OpenCV

Computer Vision Applications (T869-COMP), Reykjavik University.

## Files

| File | What it does |
| --- | --- |
| `export_models.py` | Exports yolo26n / yolo26s (or my fine-tuned `best.pt`) to ONNX with the tutorial's settings and prints the parameter count. Run in the export environment. |
| `detect_live.py` | Runs one or two ONNX models live on the camera with OpenCV DNN (letterbox 640 × 640), shows FPS, and saves frames for counting. |
| `count_pr.py` | Shows the saved frames; I type TP / FP / FN for each, and it computes precision and recall. |
| `train_colab.py` | Fine-tuning of yolo26n on my pen/pencil data (Google Colab, T4 GPU). |
| `requirements.txt` | Package versions of my inference (OpenCV) environment. |
| `requirements-export.txt` | Package versions of the export environment, from the tutorial. |

## How to run

```
# Export (separate environment)
python -m venv .export
.export/bin/python -m pip install -r requirements-export.txt
.export/bin/python export_models.py            # yolo26n + yolo26s
.export/bin/python export_models.py best.pt    # fine-tuned model

# Live detection (OpenCV environment)
python detect_live.py yolo26n.onnx yolo26s.onnx
python detect_live.py best.onnx
```

Keys in `detect_live.py`: `1` / `2` switch model, `+` / `-` change the score threshold, `b` saves the same frame for both models, `s` saves the current frame, `q` quits.

The dataset, the trained weights (`.pt`) and the ONNX files are not included.
