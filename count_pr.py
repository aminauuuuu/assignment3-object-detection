"""
Assignment 3 - Part 1, step 4: count TP / FP / FN and compute precision and recall.

Shows each saved picture (saved_frames/sceneXX_<model>.jpg) one by one.
Everything happens IN THE PICTURE WINDOW (click it first):

    type three digits   TP, then FP, then FN   (e.g. 3 0 1)
    Enter               confirm and go to the next picture
    Backspace           undo the last digit
    Esc                 stop (your answers so far are kept)

    TP = boxes labelled "cup" that are on a real cup
    FP = boxes labelled "cup" on something that is NOT a cup,
         or a second box on the same cup
    FN = real cups that have NO "cup" box

Answers are saved in counts.csv after every picture, so you can stop and
continue later. At the end it prints precision and recall for each model.

Usage:
    python count_pr.py              # class "cup"
    python count_pr.py bottle       # another class
"""
import csv
import glob
import os
import sys

import cv2

target = sys.argv[1] if len(sys.argv) > 1 else "cup"
images = sorted(glob.glob("saved_frames/scene*_*.jpg"))
if not images:
    raise SystemExit("No saved_frames/scene*.jpg pictures found. Run this inside assignment3.")

done = {}
if os.path.exists("counts.csv"):
    for row in csv.DictReader(open("counts.csv")):
        done[row["image"]] = row


def save():
    with open("counts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "model", "TP", "FP", "FN"])
        writer.writeheader()
        writer.writerows(done.values())


todo = [p for p in images if p not in done]
print(f"Counting class '{target}': {len(todo)} pictures left. Click the picture window.")

stopped = False
for number, path in enumerate(todo, 1):
    original = cv2.imread(path)
    scale = min(1.0, 1200 / original.shape[1])
    original = cv2.resize(original, None, fx=scale, fy=scale)
    digits = []
    while True:
        view = original.copy()
        labels = ["TP", "FP", "FN"]
        typed = "  ".join(f"{labels[i]}={digits[i] if i < len(digits) else '_'}" for i in range(3))
        lines = [f"{os.path.basename(path)}   ({number}/{len(todo)})",
                 f"count '{target}':  {typed}",
                 "digits, Enter = next, Backspace = undo, Esc = stop"]
        cv2.rectangle(view, (0, 0), (view.shape[1], 105), (255, 255, 255), -1)
        for i, text in enumerate(lines):
            cv2.putText(view, text, (10, 30 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0) if i < 2 else (90, 90, 90), 2)
        cv2.imshow("count", view)
        key = cv2.waitKey(50) & 0xFF
        if key == 255:
            continue
        if ord("0") <= key <= ord("9") and len(digits) < 3:
            digits.append(int(chr(key)))
        elif key in (8, 127) and digits:          # Backspace / Delete
            digits.pop()
        elif key in (13, 10) and len(digits) == 3:  # Enter
            break
        elif key == 27:                            # Esc
            stopped = True
            break
    if stopped:
        break
    model = path.rsplit("_", 1)[1].replace(".jpg", "")
    done[path] = {"image": path, "model": model,
                  "TP": digits[0], "FP": digits[1], "FN": digits[2]}
    save()
    print(f"{os.path.basename(path)}: TP={digits[0]} FP={digits[1]} FN={digits[2]}")

cv2.destroyAllWindows()
if stopped:
    print("Stopped. Run again to continue where you left off.")

print(f"\nResults for class '{target}':")
print(f"{'model':10} {'frames':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'precision':>10} {'recall':>7}")
for model in sorted({r["model"] for r in done.values()}):
    rows = [r for r in done.values() if r["model"] == model]
    tp = sum(int(r["TP"]) for r in rows)
    fp = sum(int(r["FP"]) for r in rows)
    fn = sum(int(r["FN"]) for r in rows)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    print(f"{model:10} {len(rows):>6} {tp:>4} {fp:>4} {fn:>4} {precision:>10.2f} {recall:>7.2f}")
