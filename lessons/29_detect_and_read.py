"""OCR lesson 12: the full pipeline = detector boxes + OCR text -> predictions.csv

1. take the detector's boxes (lesson 13 tiling + lesson 15 merge)
2. cut each box out of the big frame, with a margin like the teacher's crops
   (about 16% of the plate width on every side)
3. read it with our OCR model, write the text into the `text` column

Run:  uv run python lessons/29_detect_and_read.py output/pred_rfdetr_tile640_merged.csv runs/ocr_mix_v4/<date>/best.keras output/pred_full.csv
Then: python3 data/teacher_test/plate_test_v1_dedup/evaluate.py output/pred_full.csv --conf 0.72
"""

import csv
import sys

import numpy as np
from fast_plate_ocr.core.process import read_and_resize_plate_image
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.utilities.utils import load_keras_model
from PIL import Image

boxes_csv, ocr_model, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
IMAGES = "data/teacher_test/plate_test_v1/images/"
MARGIN = float(sys.argv[4]) if len(sys.argv) > 4 else 0.16
MIN_CONF = 0.3          # don't waste time reading boxes we'll throw away anyway
TMP = "output/_plate_crop.png"

cfg = load_plate_config_from_yaml("configs/ocr/uz_plate_config.yaml")
model = load_keras_model(ocr_model, cfg)

rows = list(csv.DictReader(open(boxes_csv)))
frames = {}
batch = []
for r in rows:
    if float(r["conf"]) < MIN_CONF:
        continue
    if r["image"] not in frames:
        frames = {r["image"]: Image.open(IMAGES + r["image"]).convert("RGB")}   # keep only one frame in memory
    x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
    m = MARGIN * (x2 - x1)
    frames[r["image"]].crop((x1 - m, y1 - m, x2 + m, y2 + m)).save(TMP)
    batch.append((r, read_and_resize_plate_image(TMP, cfg.img_height, cfg.img_width, cfg.image_color_mode)))

probs = model.predict(np.stack([b[1] for b in batch]), verbose=0)["plate"]
probs = probs.reshape(len(batch), cfg.max_plate_slots, cfg.vocabulary_size)
for (r, _), p in zip(batch, probs):
    r["text"] = "".join(cfg.alphabet[i] for i in p.argmax(-1)).replace(cfg.pad_char, "")

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image", "x1", "y1", "x2", "y2", "conf", "text"])
    w.writeheader()
    w.writerows(rows)
print(f"read {len(batch)} boxes -> {out_csv}")
