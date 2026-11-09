"""OCR lesson 8: type the text of real plates (fast: the model already guessed).

A window shows the plate; you answer in the TERMINAL:
  Enter        -> guess is right
  01A123BC     -> type the correct text (no spaces)
  -            -> can't read it, skip
  q            -> quit (progress is saved, run again to continue)

Output: data/ocr_real/annotations.csv (image_path, plate_text)
Run:  uv run python lessons/25_label_text.py
"""

import csv
import os
import re

import cv2

D = "data/ocr_real/"
OUT = D + "annotations.csv"
done = {}
if os.path.exists(OUT):
    done = {r["image_path"]: r["plate_text"] for r in csv.DictReader(open(OUT))}
skipped_file = D + "skipped.txt"
skipped = set(open(skipped_file).read().split()) if os.path.exists(skipped_file) else set()

todo = [r for r in csv.DictReader(open(D + "to_label.csv")) if r["image_path"] not in done and r["image_path"] not in skipped]
print(f"{len(done)} labelled, {len(skipped)} skipped, {len(todo)} to go")

for i, r in enumerate(todo, 1):
    img = cv2.imread(D + r["image_path"])
    h, w = img.shape[:2]
    cv2.imshow("plate", cv2.resize(img, (600, int(600 * h / w))))
    cv2.waitKey(300)                     # give the window time to draw before we wait for typing
    answer = input(f"[{i}/{len(todo)}] guess {r['guess']:<10} > ").strip().upper().replace(" ", "")
    if answer == "Q":
        break
    if answer == "-":
        skipped.add(r["image_path"])
        open(skipped_file, "a").write(r["image_path"] + "\n")
        continue
    text = answer or r["guess"]
    if not re.fullmatch(r"[0-9A-Z]{4,10}", text):
        print("  only A-Z and 0-9, 4-10 characters -> skipped this one, it will come back next run")
        continue
    done[r["image_path"]] = text
    with open(OUT, "w", newline="") as f:                     # save after EVERY plate
        wr = csv.writer(f)
        wr.writerow(["image_path", "plate_text"])
        wr.writerows(done.items())

cv2.destroyAllWindows()
print(f"saved {len(done)} labels -> {OUT}")
