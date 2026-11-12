"""Lesson 15: clean up tiling leftovers in a predictions.csv.

Tiling problem: a plate on the border between two tiles is cut in half, and
each tile reports its half as "a plate". The same plate is also found by the
whole-photo pass, often with a looser box. NMS doesn't remove these, because a
half box overlaps the full box only a little (IoU < 0.5).

Rule: if 70%+ of a smaller box lies inside a bigger box, they are the same
plate -> keep only the MORE CONFIDENT of the two.
(Version 1 always kept the bigger box. That removed half-plates, but often
kept a loose box instead of a tight one.)

Run:  uv run python plate/lessons/15_merge_boxes.py plate/output/pred_rfdetr_tile640.csv plate/output/pred_rfdetr_tile640_merged.csv
"""

import csv
import sys
from collections import defaultdict

src, dst = sys.argv[1], sys.argv[2]
INSIDE = 0.7
MIN_CONF = 0.3        # very unsure boxes can't remove other boxes


def area(b):
    return max((b[2] - b[0]) * (b[3] - b[1]), 1.0)     # max(): some boxes are 0 pixels wide


def share_inside(small, big):
    ix = max(0, min(small[2], big[2]) - max(small[0], big[0]))
    iy = max(0, min(small[3], big[3]) - max(small[1], big[1]))
    return ix * iy / area(small)


rows = defaultdict(list)
for r in csv.DictReader(open(src)):
    rows[r["image"]].append(r)

kept_total = dropped = 0
with open(dst, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "x1", "y1", "x2", "y2", "conf", "text"])
    writer.writeheader()
    for image, preds in rows.items():
        boxes = [[[float(p[k]) for k in ("x1", "y1", "x2", "y2")], float(p["conf"]), p, True] for p in preds]
        boxes.sort(key=lambda b: -b[1])                       # most confident first
        for a in boxes:
            if not a[3] or a[1] < MIN_CONF:
                continue
            for b in boxes:
                if b is a or not b[3]:
                    continue
                small, big = (a, b) if area(a[0]) < area(b[0]) else (b, a)
                if share_inside(small[0], big[0]) >= INSIDE:
                    b[3] = False                              # a is more confident, b goes
                    dropped += 1
        for box, conf, p, keep in boxes:
            if keep:
                writer.writerow(p)
                kept_total += 1
print(f"kept {kept_total}, dropped {dropped} pieces/copies -> {dst}")
