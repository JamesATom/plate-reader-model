"""OCR lesson 1: look at plate crops and the text a person wrote for each.

Run:  uv run python lessons/18_look_at_crops.py
"""

import csv
import random

import cv2
import numpy as np

T = "data/teacher_test/plate_test_v1/"
rows = [r for r in csv.DictReader(open(T + "crops.csv")) if r["text_ok"] == "1"]
random.seed(3)

tiles = []
for r in random.sample(rows, 24):
    crop = cv2.resize(cv2.imread(T + "crops/" + r["crop"]), (300, 80))
    caption = np.full((30, 300, 3), 255, np.uint8)
    cv2.putText(caption, r["text"], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    tiles.append(np.vstack([crop, caption]))

grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 24, 4)])
cv2.imwrite("output/18_crops.jpg", grid)
print("saved output/18_crops.jpg")
