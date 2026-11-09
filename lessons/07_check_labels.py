"""Lesson 7: draw the dataset's labels on random photos to check them.

Run:  uv run python lessons/07_check_labels.py data/uzbek_plates/train
"""

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

folder = Path(sys.argv[1])
coco = json.loads((folder / "_annotations.coco.json").read_text())

# Group boxes by photo id: {image_id: [bbox, bbox, ...]}
boxes = {}
for ann in coco["annotations"]:
    boxes.setdefault(ann["image_id"], []).append(ann["bbox"])

random.seed(1)                      # same 12 photos every run
picked = random.sample(coco["images"], 12)

tiles = []
for image in picked:
    img = cv2.imread(str(folder / image["file_name"]))
    for x, y, w, h in boxes.get(image["id"], []):
        # COCO box is x, y, width, height -> rectangle needs two corners
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 3)
    cv2.putText(img, f"id {image['id']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    tiles.append(cv2.resize(img, (320, 320)))

# 3 rows x 4 photos
grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)])
cv2.imwrite("output/07_check_labels.png", grid)
print("saved output/07_check_labels.png")
