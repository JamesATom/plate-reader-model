"""OCR lesson 13: where exactly does the full pipeline lose plates?

For every ground-truth plate with verified text:
  missed      -> detector found nothing (IoU >= 0.5, conf >= 0.72)
  misread     -> found, but text wrong: which part? how small? night?
  ok          -> found and read right

Run:  uv run python lessons/30_where_we_fail.py output/pred_full.csv
"""

import csv
import sys
from collections import Counter, defaultdict

T = "data/teacher_test/plate_test_v1/"
CONF = 0.72

preds = defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    if float(r["conf"]) >= CONF:
        preds[r["image"]].append(([float(r[k]) for k in ("x1", "y1", "x2", "y2")], r["text"]))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def size_bucket(w):
    return "<80px" if w < 80 else "80-120px" if w < 120 else "120-180px" if w < 180 else ">=180px"


def is_night(image):
    hour = int(image.split("_")[2][:2])
    return hour >= 19 or hour < 7


def which_part(truth, read):
    if len(truth) != len(read):
        return "length wrong (extra/missing char)"
    if truth[:2] != read[:2] and truth[2:] == read[2:]:
        return "region only"
    if truth[:2] == read[:2]:
        return "main part only"
    return "region + main part"


result = Counter()
by_size = defaultdict(Counter)
by_night = defaultdict(Counter)
parts = Counter()
confusions = Counter()
for g in csv.DictReader(open(T + "ground_truth.csv")):
    if g["text_ok"] != "1":
        continue
    box = [float(g[k]) for k in ("x1", "y1", "x2", "y2")]
    best = max(preds[g["image"]], key=lambda p: iou(p[0], box), default=None)
    if best is None or iou(best[0], box) < 0.5:
        status = "missed"
    elif best[1] == g["text"]:
        status = "ok"
    else:
        status = "misread"
        parts[which_part(g["text"], best[1])] += 1
        if len(best[1]) == len(g["text"]):
            for a, b in zip(g["text"], best[1]):
                if a != b:
                    confusions[f"{a}->{b}"] += 1
    result[status] += 1
    by_size[size_bucket(box[2] - box[0])][status] += 1
    by_night["night" if is_night(g["image"]) else "day"][status] += 1

n = sum(result.values())
print(f"{n} plates with verified text: ok {result['ok']}, misread {result['misread']}, missed {result['missed']}\n")
print("misread - which part is wrong:")
for k, v in parts.most_common():
    print(f"  {k:<36}{v}")
print("\nmost common letter swaps (correct->read):", ", ".join(f"{k} x{v}" for k, v in confusions.most_common(10)))
for title, table in (("plate width", by_size), ("time of day", by_night)):
    print(f"\nby {title}:")
    for k in sorted(table):
        c = table[k]
        tot = sum(c.values())
        print(f"  {k:<10} {tot:>3} plates   ok {c['ok'] / tot:>4.0%}   misread {c['misread'] / tot:>4.0%}   missed {c['missed'] / tot:>4.0%}")
