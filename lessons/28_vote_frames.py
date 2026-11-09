"""OCR lesson 11: one car = many frames -> let the frames VOTE.

In live video a tracker says "these 20-50 boxes are the same car". We read each
box, then for every character position pick the letter with the most total
confidence. One bad frame can't spoil the answer.

The teacher's test set has no tracks (repeats are parked cars, minutes or days
apart). So this is a SIMULATION: we group crops by their TRUE text, which a real
tracker would do by position. It shows the idea, it is not a fair test score.

Run:  uv run python lessons/28_vote_frames.py runs/ocr_mix_v3/<date>/best.keras
"""

import csv
import sys
from collections import defaultdict

import numpy as np
from fast_plate_ocr.core.process import read_and_resize_plate_image
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.utilities.utils import load_keras_model

T = "data/teacher_test/plate_test_v1/"
cfg = load_plate_config_from_yaml("configs/ocr/uz_plate_config.yaml")
model = load_keras_model(sys.argv[1], cfg)
rows = [r for r in csv.DictReader(open(T + "crops.csv")) if r["text_ok"] == "1"]
x = np.stack([read_and_resize_plate_image(T + "crops/" + r["crop"], cfg.img_height, cfg.img_width, cfg.image_color_mode) for r in rows])
probs = model.predict(x, verbose=0)["plate"].reshape(len(rows), cfg.max_plate_slots, cfg.vocabulary_size)


def decode(p):
    return "".join(cfg.alphabet[i] for i in p.argmax(-1)).replace(cfg.pad_char, "")


def vote(list_of_probs):
    """Add up the probabilities of all frames, slot by slot, then pick the best letter."""
    return decode(np.sum(list_of_probs, axis=0))


groups = defaultdict(list)
for r, p in zip(rows, probs):
    groups[r["text"]].append(p)

single = sum(decode(p) == r["text"] for r, p in zip(rows, probs)) / len(rows)
multi = {t: ps for t, ps in groups.items() if len(ps) > 1}
one_frame = np.mean([np.mean([decode(p) == t for p in ps]) for t, ps in multi.items()])
voted = np.mean([vote(ps) == t for t, ps in multi.items()])
print(f"every crop on its own: {single:.1%}")
print(f"{len(multi)} plates seen 2+ times: one frame {one_frame:.1%} -> all frames voting {voted:.1%}")
