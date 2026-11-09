"""OCR lesson 10: save typed labels + build a review page in the browser.

labels file: one line per crop "<index> <TEXT> [flip|rot180]" or "<index> -" (skip)
Run:  uv run python lessons/27_save_labels_and_review.py data/ocr_real2
"""

import csv
import json
import sys

from PIL import Image, ImageOps

D = sys.argv[1].rstrip("/") + "/"
rows = list(csv.DictReader(open(D + "to_label.csv")))
labels = {int(p[0]): p[1:] for p in (line.split() for line in open(D + "claude_labels.txt"))}

good = []
for i, r in enumerate(rows):
    lab = labels[i]
    if lab[0] == "-":
        continue
    if "flip" in lab:                       # mirrored -> mirror back
        ImageOps.mirror(Image.open(D + r["image_path"])).save(D + r["image_path"])
    if "rot180" in lab:                     # upside down -> turn around
        Image.open(D + r["image_path"]).rotate(180).save(D + r["image_path"])
    good.append((r["image_path"], lab[0]))

with open(D + "annotations.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_path", "plate_text"])
    w.writerows(good)
print(f"{len(good)} labelled, {len(rows) - len(good)} skipped -> {D}annotations.csv")

TEMPLATE = open("data/ocr_real/review.html").read()
start = TEMPLATE.index("const D=") + len("const D=")
end = TEMPLATE.index(";", start)
html = TEMPLATE[:start] + json.dumps(good) + TEMPLATE[end:]
html = html.replace("localStorage.getItem('st')", f"localStorage.getItem('st_{D}')").replace("localStorage.setItem('st'", f"localStorage.setItem('st_{D}'")
open(D + "review.html", "w").write(html)
print(f"review page -> {D}review.html")
