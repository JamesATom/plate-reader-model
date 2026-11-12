"""Lesson 3: preparing the plate picture before OCR reads it.

Run:  uv run python plate/lessons/03_preprocessing.py plate/data/samples/car_01.jpg
"""

import sys

import cv2
import numpy as np

img = cv2.imread(sys.argv[1])
plate = img[316:372, 543:743]      # same hand-picked box as lesson 2

# --- 1. Grayscale: 3 numbers per pixel -> 1 number --------------------------
# Colour doesn't help read letters. Black text on white is enough.
gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
print("colour plate shape:", plate.shape, "->", plate.size, "numbers")
print("gray plate shape:  ", gray.shape, "  ->", gray.size, "numbers")

# --- 2. Fixed size: every plate becomes exactly the same size ---------------
# A model always expects the same input size, like a function that only
# accepts one type. Near or far, every plate gets resized to this.
W, H = 256, 64
fixed = cv2.resize(gray, (W, H))
print("fixed size shape:  ", fixed.shape)

# --- 3. Bad real-world conditions -------------------------------------------
# Far away: the camera only captured a few pixels of the plate.
# Shrinking throws pixels away; enlarging again can NOT bring them back.
tiny = cv2.resize(gray, (40, 11))
far = cv2.resize(tiny, (W, H))

# Blurry: car moving or camera out of focus.
blurry = cv2.GaussianBlur(fixed, (15, 15), 0)

# Dark: evening or shadow. Multiply every number by 0.25.
dark = (fixed * 0.25).astype(np.uint8)

# Dark, then brightened again. Contrast stretching makes the darkest pixel 0
# and the brightest 255 - text becomes readable again.
rescued = cv2.normalize(dark, None, 0, 255, cv2.NORM_MINMAX)

# --- 4. Stack everything into one picture with labels -----------------------
panels = [
    ("1 gray, fixed size", fixed),
    ("2 far away (40x11 px)", far),
    ("3 blurry", blurry),
    ("4 dark", dark),
    ("5 dark, brightened again", rescued),
]
rows = []
for label, panel in panels:
    row = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)     # colour, so label can be red
    caption = np.full((24, W, 3), 255, dtype=np.uint8)
    cv2.putText(caption, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    rows += [caption, row]

cv2.imwrite("plate/output/03_preprocessing.png", np.vstack(rows))
print("saved plate/output/03_preprocessing.png")
