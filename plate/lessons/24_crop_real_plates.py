"""OCR lesson 7: cut real plates out of Roboflow photos + pre-fill a text guess.

Roboflow already has the plate BOXES, so we just cut them out (with some margin,
like the teacher's crops). Roboflow saved each photo 3 times (augmented copies),
so we keep only one copy per original photo.

Output: plate/data/ocr_real/images/*.png + to_label.csv (image_path, guess)
Run:  uv run python plate/lessons/24_crop_real_plates.py plate/runs/ocr_synth_v2/<date>/best.keras 300
      uv run python plate/lessons/24_crop_real_plates.py plate/runs/ocr_mix_v3/<date>/best.keras 300 2     (second batch)
"""

import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from fast_plate_ocr.core.process import read_and_resize_plate_image
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.utilities.utils import load_keras_model, postprocess_model_output
from PIL import Image

model_path, N = sys.argv[1], int(sys.argv[2])
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 1         # batch 2 = 300 NEW photos, not used in batch 1
SRC = Path("plate/data/uzbek_plates/train")
OUT = Path("plate/data/ocr_real" if BATCH == 1 else f"plate/data/ocr_real{BATCH}")
(OUT / "images").mkdir(parents=True, exist_ok=True)

coco = json.load(open(SRC / "_annotations.coco.json"))
files = {im["id"]: im["file_name"] for im in coco["images"]}

def sharpness(name, box):
    """Blurry crop -> few sharp edges -> low score (variance of the Laplacian)."""
    x, y, w, h = box
    g = cv2.cvtColor(cv2.imread(str(SRC / name))[int(y):int(y + h), int(x):int(x + w)], cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var() if g.size else 0


# Roboflow saved each photo 3 times with blur/flip/rotate: "ABC_jpg.rf.<hash>.jpg".
# Group the copies by original name "ABC", keep the SHARPEST copy.
copies = {}
for a in coco["annotations"]:
    name = files[a["image_id"]]
    copies.setdefault(name.split("_jpg.rf.")[0], []).append((name, a["bbox"]))
# Some copies are rotated 90 degrees -> the box is tall. Keep only wide (horizontal) copies.
copies = {o: [c for c in cs if c[1][2] > 2 * c[1][3]] for o, cs in copies.items()}
copies = {o: cs for o, cs in copies.items() if cs}
random.seed(0)
originals = random.sample(sorted(copies), 300)                 # batch 1 (exactly as before)
used = set(originals)
for b in range(2, BATCH + 1):                                  # batch b: NEW photos, never used before
    random.seed(b - 1)
    originals = random.sample(sorted(set(copies) - used), 300 if b < BATCH else N)
    used |= set(originals)
picked = [max(copies[o], key=lambda c: sharpness(*c)) for o in originals]

# Remember where each crop came from -> lesson 31 can re-cut it with other margins.
with open(OUT / "source.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_path", "photo", "x", "y", "w", "h"])
    w.writerows([f"images/{i:04d}.png", name, *box] for i, (name, box) in enumerate(picked))
if "--sources-only" in sys.argv:                               # old batches: don't re-cut, just save sources
    sys.exit(0)

paths = []
for i, (name, (x, y, w, h)) in enumerate(picked):
    m = random.uniform(0.1, 0.3)                               # margin around the plate
    photo = Image.open(SRC / name).convert("RGB")
    crop = photo.crop((x - w * m, y - h * m * 2, x + w * (1 + m), y + h * (1 + m * 2)))
    p = f"images/{i:04d}.png"
    crop.save(OUT / p)
    paths.append(p)

cfg = load_plate_config_from_yaml("plate/configs/ocr/uz_plate_config.yaml")
model = load_keras_model(model_path, cfg)
x = np.stack([read_and_resize_plate_image(OUT / p, cfg.img_height, cfg.img_width, cfg.image_color_mode) for p in paths])
out = model.predict(x, verbose=0)["plate"]

# Better guess = combine two models:
#   ready-made cct-s reads the MAIN part well, but often drops the region "01"
#   our v2 at least always outputs a region
from fast_plate_ocr import LicensePlateRecognizer
ready = LicensePlateRecognizer("cct-s-v2-global-model")

with open(OUT / "to_label.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_path", "guess"])
    for p, o in zip(paths, out):
        ours = postprocess_model_output(o, cfg.alphabet, cfg.max_plate_slots, cfg.vocabulary_size)[0].replace("_", "")
        main = "".join(ch for ch in ready.run(str(OUT / p))[0].plate.upper() if ch.isalnum())
        # from batch 3 on, our own model (v4+) is better than the ready-made one -> use it alone
        guess = ours if BATCH >= 3 else (main if len(main) >= 8 and main[:2].isdigit() else ours[:2] + main)
        w.writerow([p, guess])
print(f"saved {N} crops + guesses -> {OUT}/to_label.csv")
