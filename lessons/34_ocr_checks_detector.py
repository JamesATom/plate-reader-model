"""Lesson 34: let the OCR help decide "is this box really a plate?"

The detector alone says how sure it is that a box is a plate (conf).
The OCR says how sure it is about the TEXT it read (ocr_p, 0..1).
  - real plate, low detector conf, but OCR reads a clean valid plate -> keep it (fewer FN)
  - junk box (sign, grille), high detector conf, OCR can't read a plate -> drop it (fewer FP)

New score = conf^A * ocr_p^(1-A)   (A=1: detector only, A=0: OCR only)
Overlapping boxes on one plate: keep only the one with the best new score.
The new score goes into the `conf` column, so evaluate.py --conf T works as before.

First:  MIN_CONF=0.02 SAVE_OCR=1 DEDUP=0 uv run python lessons/32_read_better.py \
            output/pred_rfdetr_tile640_merged.csv output/_ocrgate.csv <models...>
Run:    uv run python lessons/34_ocr_checks_detector.py output/_ocrgate.csv output/pred_full_v11.csv 0.5
"""

import csv
import sys

src, out = sys.argv[1], sys.argv[2]
A = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5


def overlap(a, b):
    """share of the SMALLER box that lies inside the other one"""
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0           # a few detector boxes have zero size


rows = list(csv.DictReader(open(src)))
by_image = {}
for r in rows:
    r["box"] = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
    ocr = float(r["ocr_p"]) if r.get("ocr_p") else 0.0        # not read (conf < MIN_CONF) -> 0
    r["score"] = float(r["conf"]) ** A * ocr ** (1 - A)
    by_image.setdefault(r["image"], []).append(r)

# one plate = one box: best score wins, boxes mostly inside it are dropped
for boxes in by_image.values():
    boxes.sort(key=lambda r: -r["score"])
    kept = []
    for r in boxes:
        if any(overlap(r["box"], k["box"]) >= 0.5 for k in kept):
            r["score"] = 0.0
        elif r["score"] > 0:
            kept.append(r)

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "x1", "y1", "x2", "y2", "conf", "text"])
    for r in rows:
        w.writerow([r["image"], r["x1"], r["y1"], r["x2"], r["y2"], f"{r['score']:.4f}", r["text"]])
print(f"A={A}: wrote {out}")
