"""Lesson 2: a real photo is the same thing - just a bigger grid of numbers.

Run:  uv run python plate/lessons/02_real_photo.py plate/data/samples/car_01.jpg
"""

import sys

import cv2

# --- 1. Load the photo: JPEG bytes -> NumPy array ---------------------------
path = sys.argv[1]
img = cv2.imread(path)
if img is None:
    sys.exit(f"could not read {path}")

height, width, channels = img.shape
print(f"size: {width} x {height}, {channels} colour channels")
print(f"total numbers: {img.size:,}")

# --- 2. Look at a few real pixels -------------------------------------------
# img[row, col] -> row is y (top to bottom), col is x (left to right)
print("pixel at (x=100, y=650), the yellow line:", img[650, 100])
print("pixel at (x=640, y=150), the rear window:", img[150, 640])
print("pixel at (x=640, y=450), the white bumper:", img[450, 640])

# --- 3. Crop the plate - just array slicing ---------------------------------
# Coordinates picked by hand for car_01.jpg. Later the detector model
# will find these four numbers for us.
x1, y1, x2, y2 = 543, 316, 743, 372
plate = img[y1:y2, x1:x2]                 # rows first (y), then columns (x)
print("plate crop shape:", plate.shape)
cv2.imwrite("plate/output/02_plate_crop.png", plate)

# --- 4. Draw the box on a copy of the photo ---------------------------------
# This is how we'll visualise every detection the model makes.
boxed = img.copy()
cv2.rectangle(boxed, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)
cv2.putText(boxed, "plate", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2)
cv2.imwrite("plate/output/02_boxed.png", boxed)

print("saved plate/output/02_plate_crop.png and plate/output/02_boxed.png")
