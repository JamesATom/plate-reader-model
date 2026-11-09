"""OCR lesson 16: pseudo-labels = let our best models label NEW plates for us.

We still have ~2000 Roboflow photos nobody labelled. Idea:
  1. cut the plate out (sharpest copy; turn it the right way round)
  2. the ensemble (2 models) reads it
  3. keep it ONLY if the read is very sure: it fits an Uzbek plate pattern AND
     every character slot has probability >= MIN_P
  4. save 3 crops with random margins (like lesson 31) -> new training data

Mistakes in pseudo-labels teach the model wrong things, so we prefer
fewer, surer labels over many unsure ones.

Run:  uv run python lessons/33_pseudo_labels.py runs/ocr_mix_v6s/<date>/best.keras runs/ocr_mix_v6/<date>/best.keras
Out:  data/ocr_pseudo/{images/*.png, annotations.csv, all_reads.csv}, output/33_pseudo_check.jpg
"""

import csv
import importlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from fast_plate_ocr.core.process import read_and_resize_plate_image
from PIL import Image, ImageOps

sys.path.insert(0, "lessons")
rb = importlib.import_module("32_read_better")  # noqa: E402  (reuses cfg, decode_format)
from fast_plate_ocr.train.utilities.utils import load_keras_model  # noqa: E402

MIN_P = 0.95
K = 3
ROOT = Path("data/uzbek_plates")
OUT = Path("data/ocr_pseudo")
(OUT / "images").mkdir(parents=True, exist_ok=True)
TMP = "output/_pseudo.png"
random.seed(0)
cfg = rb.cfg
models = [load_keras_model(m, cfg) for m in sys.argv[1:]]

# 1. which originals did batches 1-3 already use?
used = set()
for d in ("data/ocr_real", "data/ocr_real2", "data/ocr_real3"):
    used |= {r["photo"].split("_jpg.rf.")[0] for r in csv.DictReader(open(Path(d) / "source.csv"))}


def sharpness(path, box):
    x, y, w, h = (int(v) for v in box)
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)[max(0, y):y + h, max(0, x):x + w]
    return cv2.Laplacian(g, cv2.CV_64F).var() if g.size else 0


# 2. one copy per new original: wide boxes first (not rotated), then the sharpest
picked = []
for split in ("train", "valid"):
    coco = json.load(open(ROOT / split / "_annotations.coco.json"))
    files = {im["id"]: im["file_name"] for im in coco["images"]}
    copies = {}
    for a in coco["annotations"]:
        name = files[a["image_id"]]
        copies.setdefault(name.split("_jpg.rf.")[0], []).append((ROOT / split / name, a["bbox"]))
    for orig, cs in copies.items():
        if orig in used:
            continue
        wide = [c for c in cs if c[1][2] > 1.5 * c[1][3]]
        picked.append(max(wide or cs, key=lambda c: sharpness(*c)))
print(f"{len(picked)} new photos")

FLAT = [lambda im: im, ImageOps.mirror, ImageOps.flip, lambda im: im.rotate(180)]
TALL = [lambda im: im.rotate(90, expand=True), lambda im: im.rotate(-90, expand=True),
        lambda im: ImageOps.mirror(im.rotate(90, expand=True)), lambda im: ImageOps.mirror(im.rotate(-90, expand=True))]


def cut(photo, box, turn, l, r, t, b):
    """margins l/r/t/b are shares of the plate's LONG side, in the turned plate's frame."""
    x, y, w, h = box
    long_side = max(w, h)
    # cut a generous area, turn it, then trim to the wanted margins
    big = 0.35 * long_side
    im = turn(photo.crop((x - big, y - big, x + w + big, y + h + big)))
    W, H = im.size
    pw, ph = (w, h) if turn in FLAT else (h, w)
    cx, cy = W / 2, H / 2
    return im.crop((cx - pw / 2 - l * long_side, cy - ph / 2 - t * long_side,
                    cx + pw / 2 + r * long_side, cy + ph / 2 + b * long_side))


# 3. read every orientation with the ensemble, keep the surest valid read
batch, meta = [], []
for path, box in picked:
    photo = Image.open(path).convert("RGB")
    turns = FLAT if box[2] > box[3] else TALL
    for ti, turn in enumerate(turns):
        cut(photo, box, turn, 0.16, 0.16, 0.16, 0.16).save(TMP)
        batch.append(read_and_resize_plate_image(TMP, cfg.img_height, cfg.img_width, cfg.image_color_mode))
        meta.append((path, box, ti))
x = np.stack(batch)
probs = sum(m.predict(x, verbose=0, batch_size=256)["plate"] for m in models) / len(models)
probs = probs.reshape(len(x), cfg.max_plate_slots, cfg.vocabulary_size)

best = {}
for (path, box, ti), p in zip(meta, probs):
    text, score = rb.decode_format(p)
    if path not in best or score > best[path][2]:
        best[path] = (box, ti, score, text, p)

rows, all_reads = [], []
for path, (box, ti, score, text, p) in best.items():
    slot_p = [p[s, cfg.alphabet.index(c)] for s, c in enumerate(text)] + \
             [p[s, rb.PAD] for s in range(len(text), cfg.max_plate_slots)]
    ok = min(slot_p) >= MIN_P
    all_reads.append((path.name, text, f"{min(slot_p):.3f}", int(ok)))
    if not ok:
        continue
    photo = Image.open(path).convert("RGB")
    turn = (FLAT if box[2] > box[3] else TALL)[ti]
    for k in range(K):
        m = [random.uniform(0.04, 0.30) for _ in range(4)]   # shares of the plate width
        name = f"images/{path.stem[:40]}_{k}.png"
        cut(photo, box, turn, *m).save(OUT / name)
        rows.append((name, text))

with open(OUT / "annotations.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_path", "plate_text"])
    w.writerows(rows)
with open(OUT / "all_reads.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["photo", "text", "min_slot_p", "kept"])
    w.writerows(all_reads)

# 4. contact sheet of 40 random kept plates + their labels, for a human look
kept = [r for r in rows if r[0].endswith("_0.png")]
sheet = Image.new("RGB", (4 * 256, 10 * 150), "white")
from PIL import ImageDraw  # noqa: E402
d = ImageDraw.Draw(sheet)
for i, (name, text) in enumerate(random.sample(kept, min(40, len(kept)))):
    sheet.paste(Image.open(OUT / name).resize((256, 120)), ((i % 4) * 256, (i // 4) * 150))
    d.text(((i % 4) * 256 + 5, (i // 4) * 150 + 122), text, fill="red")
sheet.save("output/33_pseudo_check.jpg")
print(f"kept {len(kept)}/{len(best)} plates (min slot p >= {MIN_P}) -> {len(rows)} crops in {OUT}")
