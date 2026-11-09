"""Lesson 6: a tiny labelling tool - draw boxes with the mouse.

How to use:
  1. drag a box around a plate with the mouse
  2. press ENTER to keep it (it is saved right away)
  3. draw the next plate, ENTER again
  4. press ESC when all plates are done

Run:  uv run python lessons/06_label_tool.py data/samples/car_01.jpg
"""

import sys
from pathlib import Path

import cv2

photo = Path(sys.argv[1])
img = cv2.imread(str(photo))
out = Path("output") / photo.with_suffix(".txt").name

WINDOW = "Draw a box, ENTER = keep, ESC = finish"
lines = []

while True:
    # Returns the box as (x, y, width, height), or all zeros after ESC.
    x, y, w, h = cv2.selectROI(WINDOW, img)
    if w == 0 or h == 0:
        break

    lines.append(f"plate {x} {y} {x + w} {y + h}")      # same format as car_01.txt
    out.write_text("\n".join(lines) + "\n")             # save after every box
    print(f"saved box {len(lines)}: {lines[-1]}")

    # Keep finished boxes visible while drawing the next one.
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.destroyAllWindows()
cv2.waitKey(1)
print(f"done - {len(lines)} boxes in {out}")
