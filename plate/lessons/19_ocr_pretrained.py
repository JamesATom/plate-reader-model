"""OCR lesson 2: try a ready-made OCR model (no training) on the teacher's crops.

Run:  uv run python plate/lessons/19_ocr_pretrained.py cct-s-v2-global-model
"""

import csv
import re
import sys

from fast_plate_ocr import LicensePlateRecognizer

T = "plate/data/teacher_test/plate_test_v1/"
model_name = sys.argv[1]
ocr = LicensePlateRecognizer(model_name)
rows = [r for r in csv.DictReader(open(T + "crops.csv")) if r["text_ok"] == "1"]


def clean(text):
    """Same as the teacher's evaluate.py: uppercase, no spaces/hyphens/dots."""
    return re.sub(r"[\s\-.]", "", text.upper())


def edit_distance(a, b):
    """How many letters must be added, removed or changed to turn a into b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


exact, chars, mistakes = 0, 0.0, []
for r in rows:
    read = clean(ocr.run(T + "crops/" + r["crop"])[0].plate)
    truth = r["text"]
    exact += read == truth
    chars += max(0.0, 1 - edit_distance(read, truth) / len(truth))
    if read != truth:
        mistakes.append((truth, read))

print(f"{model_name}: exact {exact}/{len(rows)} = {exact / len(rows):.1%}   char accuracy {chars / len(rows):.1%}")
print("some mistakes (correct -> read):")
for truth, read in mistakes[:15]:
    print(f"  {truth:<10} -> {read}")
