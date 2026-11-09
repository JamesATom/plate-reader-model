"""Lesson 8: make a smaller copy of the dataset for a quicker training run.

Full dataset = hours on a Mac. Smaller dataset = minutes. Same format.
RF-DETR wants three folders: train (study), valid (exam each epoch),
test (final exam, used once at the end).

Run:  uv run python lessons/08_make_tiny_dataset.py data/tiny_plates 200 50
      uv run python lessons/08_make_tiny_dataset.py data/medium_plates 1000 100
"""

import json
import random
import shutil
import sys
from pathlib import Path

SRC = Path("data/uzbek_plates")
DST = Path(sys.argv[1])
N_TRAIN = int(sys.argv[2])
N_EXAM = int(sys.argv[3])          # photos in valid, and the same number in test
# Separate random generators, so the exam photos stay the same
# no matter how many train photos we pick.
train_rng = random.Random(1)
exam_rng = random.Random(1)


def copy_subset(src_split, dst_split, images):
    coco = json.loads((SRC / src_split / "_annotations.coco.json").read_text())
    ids = {img["id"] for img in images}
    out = DST / dst_split
    out.mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy(SRC / src_split / img["file_name"], out / img["file_name"])
    coco["images"] = images
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in ids]
    (out / "_annotations.coco.json").write_text(json.dumps(coco))
    print(f"{dst_split}: {len(images)} photos, {len(coco['annotations'])} boxes")


shutil.rmtree(DST, ignore_errors=True)

train = json.loads((SRC / "train" / "_annotations.coco.json").read_text())["images"]
copy_subset("train", "train", train_rng.sample(train, N_TRAIN))

# The original valid folder is split in two: half exam, half final exam.
valid = json.loads((SRC / "valid" / "_annotations.coco.json").read_text())["images"]
exam_rng.shuffle(valid)
copy_subset("valid", "valid", valid[:N_EXAM])
copy_subset("valid", "test", valid[N_EXAM:2 * N_EXAM])
