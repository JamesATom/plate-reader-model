"""OCR lesson 15: read each plate better WITHOUT new training.

Three tricks, all at reading time:
  1. ensemble  - several models read the same crop, we average their probabilities
  2. TTA       - "test-time augmentation": cut the plate with 3 different margins,
                 average again (the model is sure on some cuts, unsure on others)
  3. format    - an Uzbek plate must look like one of a few patterns
                 (01 A 123 BC, 01 123 ABC, 01 H 123456, 01 + 3-7 letters).
                 Pick the pattern with the highest total probability, so a slot
                 that MUST be a digit can't come out as a letter (O vs 0, B vs 8).
  4. dedup     - two boxes on one plate -> keep the one the OCR reads most surely.

Run:  uv run python plate/lessons/32_read_better.py plate/output/pred_rfdetr_tile640_merged.csv plate/output/pred_full_v7.csv \
          plate/runs/ocr_mix_v6s/<date>/best.keras plate/runs/ocr_mix_v6/<date>/best.keras
Env:  MARGINS=0.12,0.16,0.20   FORMAT=1   DEDUP=1   (=0 turns a trick off; turns trick 3 off)
"""

import csv
import os
import sys

import numpy as np
from fast_plate_ocr.core.process import read_and_resize_plate_image
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.utilities.utils import load_keras_model
from PIL import Image

IMAGES = "plate/data/teacher_test/plate_test_v1/images/"
MARGINS = [float(m) for m in os.environ.get("MARGINS", "0.12,0.16,0.20").split(",")]
COMBINE = os.environ.get("COMBINE", "mean")         # mean | pick
DEDUP = os.environ.get("DEDUP", "1") == "1"
USE_FORMAT = os.environ.get("FORMAT", "1") == "1"
MIN_CONF = float(os.environ.get("MIN_CONF", 0.3))   # boxes below this are not read at all
SAVE_OCR = os.environ.get("SAVE_OCR", "0") == "1"   # add an ocr_p column (OCR sureness, 0..1)
TMP = "plate/output/_plate_crop.png"

cfg = load_plate_config_from_yaml("plate/configs/ocr/uz_plate_config.yaml")
A = cfg.alphabet
DIGITS = [A.index(c) for c in "0123456789"]
LETTERS = [A.index(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
PAD = A.index(cfg.pad_char)

# 9 = digit, a = letter
PATTERNS = ["99a999aa", "99999aaa", "99a999999"] + ["99" + "a" * n for n in range(3, 8)]


def decode_free(p):
    text = "".join(A[i] for i in p.argmax(-1)).replace(cfg.pad_char, "")
    return text, float(np.log(p.max(-1) + 1e-9).mean())


def decode_format(p):
    """p: (10 slots, 37) probabilities. Best text that fits one of the PATTERNS."""
    logp = np.log(p + 1e-9)
    best, best_score = None, -1e9
    for pat in PATTERNS:
        score, text = 0.0, ""
        for s in range(cfg.max_plate_slots):
            if s >= len(pat):
                score += logp[s, PAD]
                continue
            group = DIGITS if pat[s] == "9" else LETTERS
            i = group[int(logp[s, group].argmax())]
            score += logp[s, i]
            text += A[i]
        if score > best_score:
            best, best_score = text, score
    return best, best_score / cfg.max_plate_slots      # score: mean log-prob per slot (0 = fully sure)



def main(boxes_csv, out_csv, model_paths):
    models = [load_keras_model(m, cfg) for m in model_paths]
    rows = list(csv.DictReader(open(boxes_csv)))
    keep = [r for r in rows if float(r["conf"]) >= MIN_CONF]

    crops, frame_name, frame = [], None, None
    for r in keep:
        if r["image"] != frame_name:
            frame_name, frame = r["image"], Image.open(IMAGES + r["image"]).convert("RGB")
        x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        for mg in MARGINS:
            m = mg * (x2 - x1)
            frame.crop((x1 - m, y1 - m, x2 + m, y2 + m)).save(TMP)
            crops.append(read_and_resize_plate_image(TMP, cfg.img_height, cfg.img_width, cfg.image_color_mode))
    x = np.stack(crops)

    shape = (len(keep), len(MARGINS), cfg.max_plate_slots, cfg.vocabulary_size)
    per_model = np.stack([m.predict(x, verbose=0)["plate"].reshape(shape).mean(1) for m in models])  # avg margins
    decode = decode_format if USE_FORMAT else decode_free

    for j, r in enumerate(keep):
        if COMBINE == "pick":      # whole text from the single model that is surest about this plate
            r["text"], r["ocr"] = max((decode(pm[j]) for pm in per_model), key=lambda t: t[1])
        else:                      # "mean": average the models' probabilities slot by slot
            r["text"], r["ocr"] = decode(per_model[:, j].mean(0))


    # trick 4: two boxes on ONE plate (e.g. full plate + a box that lost the region "01").
    # Keep the box whose text the OCR reads most surely; it inherits the higher detector conf.
    def overlap(a, b):
        """share of the SMALLER box that lies inside the other one"""
        ax1, ay1, ax2, ay2 = (float(a[k]) for k in ("x1", "y1", "x2", "y2"))
        bx1, by1, bx2, by2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
        inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
        return inter / min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))


    if DEDUP:
        for a in keep:
            for b in keep:
                if a is b or a["image"] != b["image"] or float(a["conf"]) == 0 or float(b["conf"]) == 0:
                    continue
                if overlap(a, b) >= 0.5:
                    win, lose = (a, b) if a["ocr"] >= b["ocr"] else (b, a)
                    win["conf"] = f"{max(float(a['conf']), float(b['conf'])):.4f}"
                    lose["conf"] = "0"
    for r in rows:
        ocr = r.pop("ocr", None)
        if SAVE_OCR:
            r["ocr_p"] = f"{np.exp(ocr):.4f}" if ocr is not None else ""

    with open(out_csv, "w", newline="") as f:
        cols = ["image", "x1", "y1", "x2", "y2", "conf", "text"] + (["ocr_p"] if SAVE_OCR else [])
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(models)} models x {len(MARGINS)} margins, format={USE_FORMAT}: read {len(keep)} boxes -> {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
