"""
Assignment 3 - Part 1 (and Part 2 step 8): run a YOLO26 ONNX model live in OpenCV.

Run this in your normal .opencv environment (the one from Assignments 1 and 2).

Usage:
    python detect_live.py yolo26n.onnx yolo26s.onnx          # compare two models
    python detect_live.py best.onnx --classes best_classes.json   # your fine-tuned model
    python detect_live.py yolo26n.onnx --cam 1               # use another camera (e.g. iPhone)

Keys:
    1 / 2   switch between the first and second model
    + / -   raise / lower the score threshold (0.05 steps)
    s       save the current frame (with boxes) for counting precision/recall
    b       save BOTH models on the SAME frame (best for Task 4)
    q       quit
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

SIZE = 640  # the network wants 640 x 640 images


# ---------------------------------------------------------------- helpers
def letterbox(image, size=SIZE):
    """Shrink the image to fit in a size x size square, fill the rest with grey.
    Returns the square image plus the numbers needed to undo it later."""
    height, width = image.shape[:2]
    gain = min(size / width, size / height)                 # how much we shrink
    new_width, new_height = round(width * gain), round(height * gain)
    resized = cv2.resize(image, (new_width, new_height))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    padded = cv2.copyMakeBorder(
        resized, top, size - new_height - top, left, size - new_width - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, gain, left, top


def detect(net, frame, threshold):
    """Run the network on one frame. Returns rows [x1, y1, x2, y2, score, class_id]
    in the coordinates of the ORIGINAL frame."""
    height, width = frame.shape[:2]
    padded, gain, left, top = letterbox(frame)
    # pixels 0..255 -> 0..1, BGR -> RGB, shape (1, 3, 640, 640)
    blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()                      # shape (1, 300, 6), no NMS needed
    detections = output[0][output[0, :, 4] >= threshold].copy()
    # undo the letterbox: remove the grey bar, then scale back up
    detections[:, [0, 2]] = ((detections[:, [0, 2]] - left) / gain).clip(0, width)
    detections[:, [1, 3]] = ((detections[:, [1, 3]] - top) / gain).clip(0, height)
    return detections


def draw(frame, detections, class_names):
    big = frame.shape[1] / 1000          # bigger text and lines on big camera images
    for x1, y1, x2, y2, score, class_id in detections:
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(class_id)
        name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (0, 200, 0), max(2, int(3 * big)))
        label = f"{name} {score:.2f}"
        cv2.putText(frame, label, (p1[0], max(p1[1] - int(10 * big), int(30 * big))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * big, (0, 200, 0), max(2, int(2 * big)))


def load_classes(onnx_path, classes_arg):
    path = Path(classes_arg) if classes_arg else Path(onnx_path).with_name(
        Path(onnx_path).stem + "_classes.json")
    if path.exists():
        return json.load(open(path))
    print(f"(no class file {path}, showing class numbers)")
    return []


# ---------------------------------------------------------------- main
parser = argparse.ArgumentParser()
parser.add_argument("models", nargs="+", help="one or two .onnx files")
parser.add_argument("--classes", help="class-name .json (default: <model>_classes.json)")
parser.add_argument("--cam", type=int, default=0)
parser.add_argument("--threshold", type=float, default=0.35)
args = parser.parse_args()

models = []
for path in args.models[:2]:
    net = cv2.dnn.readNetFromONNX(path)   # OpenCV's built-in engine (default)
    models.append((Path(path).stem, net, load_classes(path, args.classes)))
    print("loaded", path)

cap = cv2.VideoCapture(args.cam)
if not cap.isOpened():
    raise SystemExit("Could not open the camera. Try --cam 1")

os.makedirs("saved_frames", exist_ok=True)
log = open("saved_frames/log.csv", "a", newline="")
writer = csv.writer(log)

current = 0
threshold = args.threshold
fps = 0.0
previous = time.perf_counter()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    name, net, class_names = models[current]
    raw = frame.copy()          # clean copy, without boxes

    t0 = time.perf_counter()
    detections = detect(net, frame, threshold)
    infer_ms = (time.perf_counter() - t0) * 1000

    draw(frame, detections, class_names)

    # FPS, smoothed so the number doesn't jump around
    now = time.perf_counter()
    fps = 0.9 * fps + 0.1 * (1 / (now - previous)) if fps else 1 / (now - previous)
    previous = now

    info = f"{name}  FPS {fps:.1f}  net {infer_ms:.0f} ms  thr {threshold:.2f}  boxes {len(detections)}"
    big = frame.shape[1] / 1000
    cv2.putText(frame, info, (10, int(40 * big)), cv2.FONT_HERSHEY_SIMPLEX,
                0.9 * big, (0, 0, 255), max(2, int(2 * big)))
    cv2.imshow("YOLO26 in OpenCV", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key in (ord("1"), ord("2")) and int(chr(key)) <= len(models):
        current = int(chr(key)) - 1
        fps = 0.0
    elif key in (ord("+"), ord("=")):
        threshold = min(threshold + 0.05, 0.95)
    elif key == ord("-"):
        threshold = max(threshold - 0.05, 0.05)
    elif key == ord("s"):
        stamp = time.strftime("%H%M%S") + f"_{int(now * 1000) % 1000:03d}"
        filename = f"saved_frames/{name}_{stamp}.jpg"
        cv2.imwrite(filename, frame)
        found = [class_names[int(c)] if int(c) < len(class_names) else int(c)
                 for c in detections[:, 5]]
        writer.writerow([filename, name, f"{threshold:.2f}", " ".join(map(str, found))])
        log.flush()
        print("saved", filename, found)
    elif key == ord("b"):
        scene = len([r for r in open("saved_frames/log.csv")]) // max(len(models), 1) + 1
        for m_name, m_net, m_classes in models:
            image = raw.copy()
            dets = detect(m_net, image, threshold)
            draw(image, dets, m_classes)
            filename = f"saved_frames/scene{scene:02d}_{m_name}.jpg"
            cv2.imwrite(filename, image)
            found = [m_classes[int(c)] if int(c) < len(m_classes) else int(c)
                     for c in dets[:, 5]]
            writer.writerow([filename, m_name, f"{threshold:.2f}", " ".join(map(str, found))])
            log.flush()
            print("saved", filename, found)

cap.release()
log.close()
cv2.destroyAllWindows()
