"""Lesson 1: an image is just numbers.

numbers -> PNG file -> image you can see -> numbers again
"""

import os

import cv2
import numpy as np

# --- 1. Create a tiny 2x2 image from numbers --------------------------------
# Shape is (height, width, 3). Each pixel is 3 numbers from 0 to 255.
# dtype=uint8 means "unsigned 8-bit integer": exactly the 0..255 range.
#
# Remember: OpenCV uses B, G, R order, not R, G, B.
img = np.array(
    [
        [[0, 0, 255], [0, 255, 0]],        # row 0: red,  green
        [[255, 0, 0], [0, 255, 255]],      # row 1: blue, yellow
    ],
    dtype=np.uint8,
)

print("shape:", img.shape)   # (2, 2, 3) -> 2 rows, 2 columns, 3 colours
print("pixel at row 0, col 0 (B, G, R):", img[0, 0])

# --- 2. Enlarge it so it's visible, then save as PNG ------------------------
# 2x2 pixels would be invisible on screen. INTER_NEAREST copies each pixel
# into a 200x200 block instead of blending colours together.
big = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("plate/output/01_squares.png", big)
print("saved plate/output/01_squares.png - open it in Preview")

# --- 3. Load the file back: compressed bytes -> numbers ---------------------
loaded = cv2.imread("plate/output/01_squares.png")
print("loaded shape:", loaded.shape)              # (400, 400, 3)
print("top-left pixel:", loaded[0, 0])            # red again: [0 0 255]
print("bottom-right pixel:", loaded[399, 399])    # yellow:    [0 255 255]
print("total numbers in this image:", loaded.size)

# --- 4. Compare size on disk vs size in memory ------------------------------
print("bytes on disk (PNG, compressed):", os.path.getsize("plate/output/01_squares.png"))
print("bytes in memory (raw numbers):  ", loaded.nbytes)
