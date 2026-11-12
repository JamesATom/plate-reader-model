"""Lesson 17: make a copy of the teacher's test set where each plate is labelled once.

Some plates have two "must find" boxes (same text, overlapping). One correct
detection can match only one of them, so the other always counts as missed.
Fix: keep the tighter (smaller) box as "must find", turn the other into
ignore=1 - a detection there is then neither a hit nor a mistake.

The teacher's original folder is NOT changed. The copy shares the photos
(symlink) and has its own ground_truth.csv + evaluate.py.

Run:  uv run python plate/lessons/17_dedup_ground_truth.py
"""

import csv
import shutil
from collections import defaultdict
from pathlib import Path

SRC = Path("plate/data/teacher_test/plate_test_v1")
DST = Path("plate/data/teacher_test/plate_test_v1_dedup")


def box(r):
    return [float(r[k]) for k in ("x1", "y1", "x2", "y2")]


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(area(a) + area(b) - inter, 1.0)


DST.mkdir(parents=True, exist_ok=True)
shutil.copy(SRC / "evaluate.py", DST / "evaluate.py")
if not (DST / "images").exists():
    (DST / "images").symlink_to((SRC / "images").resolve())

rows = list(csv.DictReader(open(SRC / "ground_truth.csv")))
by_image = defaultdict(list)
for r in rows:
    if r["ignore"] == "0":
        by_image[r["image"]].append(r)

changed = 0
for image, plates in by_image.items():
    for i, a in enumerate(plates):
        for b in plates[i + 1:]:
            same_text = a["text"] and a["text"] == b["text"]
            if a["ignore"] == "0" and b["ignore"] == "0" and (same_text or iou(box(a), box(b)) > 0.3):
                bigger = a if area(box(a)) > area(box(b)) else b
                bigger["ignore"] = "1"
                changed += 1
                print(f"{image}  {a['text']}: kept {[int(v) for v in box(a if bigger is b else b)]}, ignored {[int(v) for v in box(bigger)]}")

with open(DST / "ground_truth.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"{changed} duplicate boxes turned into ignore -> {DST}/ground_truth.csv")
