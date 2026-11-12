"""Lesson 16: picture sheets of every false box (FP) and every missed plate (FN).

Same matching rules as the teacher's evaluate.py. Each tile is numbered, so we
can talk about "FP 7" or "FN 12".
  FP sheet: red = our wrong box
  FN sheet: green = the plate we missed, red = our closest box (if any)

Run:  uv run python plate/lessons/16_review_errors.py plate/output/pred_rfdetr_tile640_merged.csv 0.7
"""

import csv
import sys
from collections import defaultdict

import cv2
import numpy as np

T = "plate/data/teacher_test/plate_test_v1/"
pred_csv, conf_min = sys.argv[1], float(sys.argv[2])
W, H = 360, 240


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1.0)


gt, ign = defaultdict(list), defaultdict(list)
for r in csv.DictReader(open(T + "ground_truth.csv")):
    box = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
    (ign if r["ignore"] == "1" else gt)[r["image"]].append(box)

preds = defaultdict(list)
for r in csv.DictReader(open(pred_csv)):
    if float(r["conf"]) >= conf_min:
        preds[r["image"]].append((float(r["conf"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))

fps, fns = [], []
for img in set(gt) | set(preds):
    used = set()
    for conf, box in sorted(preds[img], key=lambda p: -p[0]):
        best, bi = 0.0, None
        for i, g in enumerate(gt[img]):
            if i not in used and iou(box, g) > best:
                best, bi = iou(box, g), i
        if best >= 0.5:
            used.add(bi)
        elif not any(iou(box, g) >= 0.5 for g in ign[img]):
            fps.append((img, box, f"conf {conf:.2f}"))
    for i, g in enumerate(gt[img]):
        if i not in used:
            near = max(preds[img], key=lambda p: iou(p[1], g), default=None)
            near_box = near[1] if near and iou(near[1], g) > 0 else None
            fns.append((img, g, near_box))


def tile(img, focus, red, green, text):
    frame = cv2.imread(T + "images/" + img)
    x1, y1, x2, y2 = focus
    pad = int(max(x2 - x1, (y2 - y1) * 2, 80))
    cx1, cy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
    cx2, cy2 = min(frame.shape[1], int(x2) + pad), min(frame.shape[0], int(y2) + pad)
    for box, color in ((green, (0, 220, 0)), (red, (0, 0, 255))):
        if box:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    t = cv2.resize(frame[cy1:cy2, cx1:cx2], (W, H))
    cv2.rectangle(t, (0, H - 22), (W, H), (0, 0, 0), -1)
    cv2.putText(t, text, (4, H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return t


def sheet(tiles, path):
    while len(tiles) % 6:
        tiles.append(np.zeros((H, W, 3), np.uint8))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))


fps.sort(key=lambda f: f[0]); fns.sort(key=lambda f: f[0])
sheet([tile(img, b, b, None, f"FP{n} {txt} w{int(b[2]-b[0])}") for n, (img, b, txt) in enumerate(fps, 1)],
      "plate/output/16_fp.jpg")
sheet([tile(img, g, near, g, f"FN{n} w{int(g[2]-g[0])} {img[:10]}") for n, (img, g, near) in enumerate(fns, 1)],
      "plate/output/16_fn.jpg")
with open("plate/output/16_errors.csv", "w") as f:
    f.write("id,image,x1,y1,x2,y2\n")
    for n, (img, b, _) in enumerate(fps, 1): f.write(f"FP{n},{img},{','.join(str(int(v)) for v in b)}\n")
    for n, (img, b, _) in enumerate(fns, 1): f.write(f"FN{n},{img},{','.join(str(int(v)) for v in b)}\n")
print(f"{len(fps)} FP -> plate/output/16_fp.jpg, {len(fns)} FN -> plate/output/16_fn.jpg")
