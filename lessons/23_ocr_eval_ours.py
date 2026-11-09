"""OCR lesson 6: score OUR trained OCR model on the teacher's real crops.

Run:  uv run python lessons/23_ocr_eval_ours.py runs/ocr_synth_v1/<date>/best.keras
"""

import csv
import importlib
import sys

import numpy as np
from fast_plate_ocr.core.process import read_and_resize_plate_image
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.utilities.utils import load_keras_model, postprocess_model_output

sys.path.insert(0, "lessons")
edit_distance = importlib.import_module("19_ocr_pretrained_helpers").edit_distance

T = "data/teacher_test/plate_test_v1/"
cfg = load_plate_config_from_yaml("configs/ocr/uz_plate_config.yaml")
model = load_keras_model(sys.argv[1], cfg)
rows = [r for r in csv.DictReader(open(T + "crops.csv")) if r["text_ok"] == "1"]

images = np.stack([
    read_and_resize_plate_image(T + "crops/" + r["crop"], cfg.img_height, cfg.img_width, cfg.image_color_mode)
    for r in rows
])
outputs = model.predict(images, verbose=0)
if isinstance(outputs, dict):
    outputs = outputs["plate"]

exact, chars, region_ok, mistakes = 0, 0.0, 0, []
for r, out in zip(rows, outputs):
    read, _ = postprocess_model_output(out, cfg.alphabet, cfg.max_plate_slots, cfg.vocabulary_size)
    read, truth = read.replace(cfg.pad_char, ""), r["text"]
    exact += read == truth
    region_ok += read[:2] == truth[:2]
    chars += max(0.0, 1 - edit_distance(read, truth) / len(truth))
    if read != truth:
        mistakes.append((truth, read))

n = len(rows)
print(f"exact {exact}/{n} = {exact / n:.1%}   char accuracy {chars / n:.1%}   region code right {region_ok / n:.1%}")
print("some mistakes (correct -> read):")
for truth, read in mistakes[:15]:
    print(f"  {truth:<10} -> {read}")
