"""OCR lesson 14: cut every real plate 5 times with DIFFERENT margins.

Lesson 30-ish finding: the OCR got 5-9% worse when the crop margin changed a
little. Real detector boxes are never exactly the same, so the model must not
care how tight or loose the crop is. We re-cut each labelled plate from its
original photo with random margins on each side (and a small shift).

Some saved crops were turned the right way round by hand (mirror / 180 deg).
The original photo is still turned, so we try 4 orientations and let our OCR
model pick: the orientation where it reads the label (or is most confident).

Output: data/ocr_real_jit/{images/*.png, annotations.csv}
Run:  uv run python lessons/31_jitter_crops.py runs/ocr_mix_v4/<date>/best.keras data/ocr_real data/ocr_real2 data/ocr_real3
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
from fast_plate_ocr.core.process import read_and_resize_plate_image
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.utilities.utils import load_keras_model
from PIL import Image, ImageOps

SRC = Path("data/uzbek_plates/train")
OUT = Path("data/ocr_real_jit")
K = 5
(OUT / "images").mkdir(parents=True, exist_ok=True)
random.seed(0)

TURNS = [lambda im: im, ImageOps.mirror, ImageOps.flip, lambda im: im.rotate(180)]


cfg = load_plate_config_from_yaml("configs/ocr/uz_plate_config.yaml")
model = load_keras_model(sys.argv[1], cfg)
TMP = "output/_turn.png"


def best_turn(photo, x, y, w, h, label):
    """Read the plate in all 4 orientations; keep the one that gives the label."""
    m = 0.16 * w
    imgs = []
    for t in TURNS:
        t(photo.crop((x - m, y - m, x + w + m, y + h + m))).save(TMP)
        imgs.append(read_and_resize_plate_image(TMP, cfg.img_height, cfg.img_width, cfg.image_color_mode))
    p = model.predict(np.stack(imgs), verbose=0)["plate"].reshape(4, cfg.max_plate_slots, cfg.vocabulary_size)
    reads = ["".join(cfg.alphabet[i] for i in q.argmax(-1)).replace(cfg.pad_char, "") for q in p]
    if label in reads:
        return TURNS[reads.index(label)]
    return TURNS[int(p.max(-1).mean(-1).argmax())]


rows_out = []
for d in sys.argv[2:]:
    d = Path(d)
    labels = {r["image_path"]: r["plate_text"] for r in csv.DictReader(open(d / "annotations.csv"))}
    for s in csv.DictReader(open(d / "source.csv")):
        if s["image_path"] not in labels:
            continue                                            # skipped (unreadable)
        photo = Image.open(SRC / s["photo"]).convert("RGB")
        x, y, w, h = (float(s[k]) for k in ("x", "y", "w", "h"))
        turn = best_turn(photo, x, y, w, h, labels[s["image_path"]])
        for k in range(K):
            # margins as a share of plate WIDTH, like the teacher's crops (~0.16 each side)
            l, r, t, b = (random.uniform(0.04, 0.30) * w for _ in range(4))
            crop = turn(photo.crop((x - l, y - t, x + w + r, y + h + b)))
            name = f"images/{d.name}_{Path(s['image_path']).stem}_{k}.png"
            crop.save(OUT / name)
            rows_out.append((name, labels[s["image_path"]]))

with open(OUT / "annotations.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["image_path", "plate_text"])
    wr.writerows(rows_out)
print(f"{len(rows_out)} crops ({len(rows_out) // K} plates x {K}) -> {OUT}")
