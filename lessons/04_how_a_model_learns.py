"""Lesson 4: how a model learns - with a model that has only ONE number.

Run:  uv run python lessons/04_how_a_model_learns.py data/samples/car_01.jpg
"""

import sys

import cv2
import numpy as np

img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img[316:372, 543:743], cv2.COLOR_BGR2GRAY)

# --- 1. Labeled examples: the "answer key" ----------------------------------
# A human looked at the plate and marked some pixels: (x, y) -> letter or not.
# For a real model, this is thousands of labeled photos.
LETTER = [(80, 16), (80, 24), (110, 28), (130, 28), (140, 40), (165, 40)]
BACKGROUND = [(75, 12), (100, 12), (85, 36), (145, 28), (170, 24), (120, 44)]

examples = [(gray[y, x], "letter") for x, y in LETTER] + \
           [(gray[y, x], "background") for x, y in BACKGROUND]

# --- 2. The model: one rule with one adjustable number t --------------------
def model(pixel, t):
    return "letter" if pixel < t else "background"


def count_mistakes(t):
    return sum(model(pixel, t) != answer for pixel, answer in examples)


# --- 3. Training: try values for t, count mistakes on the examples ----------
print("  t   mistakes")
results = {}
for t in range(0, 256, 10):
    results[t] = count_mistakes(t)
    print(f"{t:3}   {results[t]:2}  {'#' * results[t]}")

# Pick the middle of the values with the fewest mistakes.
fewest = min(results.values())
good = [t for t, m in results.items() if m == fewest]
t = good[len(good) // 2]
print(f"learned t = {t}")

# --- 4. Use the trained model on ALL 11,200 pixels --------------------------
# It learned from 12 pixels, but now decides for every pixel - also ones it
# never saw. Letter -> black (0), background -> white (255).
result = np.where(gray < t, 0, 255).astype(np.uint8)

# --- 5. Picture: the labeled examples on top, the model's answer below ------
big = cv2.cvtColor(cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST),
                   cv2.COLOR_GRAY2BGR)
for x, y in LETTER:
    cv2.circle(big, (x * 4 + 2, y * 4 + 2), 6, (0, 200, 0), -1)      # green = letter
for x, y in BACKGROUND:
    cv2.circle(big, (x * 4 + 2, y * 4 + 2), 6, (0, 0, 255), -1)      # red = background
answer = cv2.cvtColor(cv2.resize(result, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST),
                      cv2.COLOR_GRAY2BGR)
cv2.imwrite("output/04_learning.png", np.vstack([big, answer]))
print("saved output/04_learning.png")
