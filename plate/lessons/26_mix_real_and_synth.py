"""OCR lesson 9: mix synthetic + real plates into one training set.

288 real plates vs 30,000 synthetic: the real ones would get lost, so we
REPEAT each real plate many times (oversampling). Augmentation makes every
repeat look a little different.

Some cars appear twice (two photos of the same plate). We split by plate TEXT,
so the same car is never in both train and val.

Run:  uv run python plate/lessons/26_mix_real_and_synth.py                                   (v3)
      uv run python plate/lessons/26_mix_real_and_synth.py ocr_synth_01 ocr_mix_v4 ocr_real ocr_real2   (v4)
"""

import csv
import os
import random
import sys

REPEAT = int(os.environ.get("REPEAT", 30))      # v6: crops already x5 -> REPEAT=6
ROOT = "plate/data/"


def read(csv_dir):
    return [("../" + csv_dir + r["image_path"], r["plate_text"]) for r in csv.DictReader(open(ROOT + csv_dir + "annotations.csv"))]


SYNTH = sys.argv[1] if len(sys.argv) > 1 else "ocr_synth"
OUT = sys.argv[2] if len(sys.argv) > 2 else "ocr_mix"
REAL = sys.argv[3:] or ["ocr_real"]
real = [r for d in REAL for r in read(d + "/")]
texts = sorted({t for _, t in real})
random.seed(0)
val_texts = set(random.sample(texts, len(texts) // 7))     # ~15% of plates for val
real_train = [r for r in real if r[1] not in val_texts]
real_val = [r for r in real if r[1] in val_texts]

train = read(SYNTH + "/train/") + real_train * REPEAT
val = read(SYNTH + "/val/")[:300] + real_val
random.shuffle(train)

os.makedirs(ROOT + OUT, exist_ok=True)
for name, rows in (("train", train), ("val", val), ("val_real", real_val)):
    with open(ROOT + f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "plate_text"])
        w.writerows(rows)
print(f"real: {len(real_train)} train (x{REPEAT}), {len(real_val)} val | train {len(train)}, val {len(val)}")
