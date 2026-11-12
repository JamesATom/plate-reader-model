"""Lesson 5: a label is just a box written as 4 numbers.

Each photo has a text file with the same name next to it:
    plate/data/samples/car_01.jpg   <- the photo
    plate/data/samples/car_01.txt   <- where the plates are

Run:  uv run python plate/lessons/05_labels.py plate/data/samples/car_01.jpg
"""

import sys
from pathlib import Path

import cv2

photo = Path(sys.argv[1])
labels = photo.with_suffix(".txt")      # car_01.jpg -> car_01.txt

img = cv2.imread(str(photo))

for line in labels.read_text().splitlines():
    if not line.strip() or line.startswith("#"):    # skip empty lines and comments
        continue
    name, x1, y1, x2, y2 = line.split()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    print(f"{name}: top-left ({x1}, {y1}), bottom-right ({x2}, {y2}), size {x2 - x1}x{y2 - y1}")

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, name, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("plate/output/05_labels.png", img)
print("saved plate/output/05_labels.png")
