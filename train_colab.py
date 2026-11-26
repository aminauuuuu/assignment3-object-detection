"""
Assignment 3 - Part 2: fine-tune yolo26n on my pen/pencil dataset.
Run on Google Colab with a T4 GPU (these were the notebook cells).

Dataset: exported from Roboflow in YOLO26 format (not included in this repo).
"""
# Cell 1 (in Colab):  !pip install -q ultralytics==8.4.27
# Cell 2 (in Colab):  !unzip -q -o /content/*.zip -d /content/data

import yaml
from ultralytics import YOLO

# Fix the dataset paths written by Roboflow
d = yaml.safe_load(open('/content/data/data.yaml'))
d['path'] = '/content/data'
d['train'] = 'train/images'
d['val'] = 'valid/images'
d.pop('test', None)
yaml.safe_dump(d, open('/content/data/data.yaml', 'w'))
print(d)  # names: ['pen', 'pencil'] -> 0 = pen, 1 = pencil

# Fine-tune the pretrained yolo26n (batch 4: only 16 training images)
model = YOLO('yolo26n.pt')
model.train(data='/content/data/data.yaml', epochs=300, imgsz=640,
            batch=4, patience=50, name='final')

# Validate the best weights: precision, recall, F1, mAP50 and confusion matrix
best = YOLO('/content/runs/detect/final/weights/best.pt')
m = best.val(data='/content/data/data.yaml', batch=4, name='final_val')
P, R = m.box.mp, m.box.mr
print(f"precision={P:.2f} recall={R:.2f} F1={2*P*R/(P+R):.2f} mAP50={m.box.map50:.3f}")
