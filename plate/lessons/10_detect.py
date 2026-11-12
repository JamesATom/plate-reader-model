"""Lesson 10: use the trained detector on a new photo.

Run:  uv run python plate/lessons/10_detect.py plate/data/samples/car_01.jpg
"""

import sys
from pathlib import Path

import cv2
from rfdetr import RFDETRNano

# Load the "knowledge" file we got from training.
model = RFDETRNano(pretrain_weights="plate/runs/rfdetr_tiny/checkpoint_best_total.pth")

photo = Path(sys.argv[1])
img = cv2.imread(str(photo))
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # the model expects R, G, B

# threshold: only keep boxes the model is at least 30% sure about
found = model.predict(rgb, threshold=0.3)

for (x1, y1, x2, y2), confidence in zip(found.xyxy.astype(int), found.confidence):
    print(f"plate {x1} {y1} {x2} {y2}   confidence {confidence:.0%}")
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{confidence:.0%}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

print(f"found {len(found)} plates")
cv2.imwrite(f"plate/output/10_detect_{photo.stem}.png", img)
print(f"saved plate/output/10_detect_{photo.stem}.png")
