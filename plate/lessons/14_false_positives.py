"""Lesson 14: cut out every false box so we can look at them.

A false box = a prediction that matches no "must find" plate and no "ignore" region.
Uses the same matching rules as the teacher's evaluate.py.

Run:  uv run python plate/lessons/14_false_positives.py plate/output/pred_rfdetr_tile640.csv 0.7
"""

import csv
import sys
from collections import defaultdict

import cv2
import numpy as np

T = "plate/data/teacher_test/plate_test_v1/"
pred_csv, conf_min = sys.argv[1], float(sys.argv[2])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


gt, ign = defaultdict(list), defaultdict(list)
for r in csv.DictReader(open(T + "ground_truth.csv")):
    box = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
    (ign if r["ignore"] == "1" else gt)[r["image"]].append(box)

preds = defaultdict(list)
for r in csv.DictReader(open(pred_csv)):
    if float(r["conf"]) >= conf_min:
        preds[r["image"]].append((float(r["conf"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))

false = []
for img, boxes in preds.items():
    used = set()
    for conf, box in sorted(boxes, key=lambda p: -p[0]):          # highest confidence first, like evaluate.py
        best, bi = 0.0, None
        for i, g in enumerate(gt[img]):
            if i not in used and iou(box, g) > best:
                best, bi = iou(box, g), i
        if best >= 0.5:
            used.add(bi)
        elif not any(iou(box, g) >= 0.5 for g in ign[img]):
            false.append((img, conf, box, best))
print(f"{len(false)} false boxes")

# One tile per false box: the box (red) with some surroundings, plus confidence.
tiles = []
for img, conf, (x1, y1, x2, y2), best in sorted(false, key=lambda f: -f[1]):
    frame = cv2.imread(T + "images/" + img)
    pad = int(max(x2 - x1, y2 - y1))
    cx1, cy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
    crop = frame[cy1:int(y2) + pad, cx1:int(x2) + pad].copy()
    cv2.rectangle(crop, (int(x1) - cx1, int(y1) - cy1), (int(x2) - cx1, int(y2) - cy1), (0, 0, 255), 2)
    tile = cv2.resize(crop, (240, 160))
    label = f"{conf:.2f} iou{best:.1f} {int(x2 - x1)}px"
    cv2.putText(tile, label, (4, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    tiles.append(tile)

while len(tiles) % 8:
    tiles.append(np.zeros_like(tiles[0]))
sheet = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])
cv2.imwrite("plate/output/14_false_boxes.jpg", sheet)
print("saved plate/output/14_false_boxes.jpg")
