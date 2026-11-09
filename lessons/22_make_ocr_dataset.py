"""OCR lesson 5: many plate types + save a training dataset (image + text).

Types (share of the dataset):
  white private   01 A 123 BC     60%
  white company   01 123 ABC      15%
  yellow          01 H 011176     8%
  green           01 M 020402     5%
  personalized    70 AHMAD        12%   (legal from 2026)

Output: data/ocr_synth/{train,val}/images/*.png + annotations.csv (image_path, plate_text)
Run:  uv run python lessons/22_make_ocr_dataset.py 30000 1000 data/ocr_synth_01
      uv run python lessons/22_make_ocr_dataset.py 30000 1000 data/ocr_synth_hard hard
"""

import csv
import importlib
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "lessons")
make_it_real = importlib.import_module("21_make_it_real").make_it_real

F = "/System/Library/Fonts/Supplemental/"
FONTS = [F + "Arial Narrow Bold.ttf", F + "Arial Bold.ttf", F + "Arial Narrow.ttf"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVXYZ"
DIGITS = "0123456789"
REGIONS = ["01", "10", "20", "25", "30", "40", "50", "60", "70", "75", "80", "85", "90", "95"]
NAMES = ["AHMAD", "BOBUR", "AZIZ", "LAYLO", "TIMUR", "SARDOR", "MALIKA", "JASUR", "UMID", "NODIR", "KAMOL", "ZARINA"]


def digits(k):
    return "".join(random.choices(DIGITS, k=k))


def letters(k):
    return "".join(random.choices(LETTERS, k=k))


def random_plate():
    """Returns (region, parts to draw, background color)."""
    kind = random.choices(["private", "company", "yellow", "green", "personal"], [60, 15, 8, 5, 12])[0]
    # Tashkent cameras see mostly "01" -> give it 40%, the rest share 60%
    region = "01" if random.random() < 0.4 else random.choice(REGIONS)
    if kind == "private":
        return region, [letters(1), digits(3), letters(2)], "white"
    if kind == "company":
        return region, [digits(3), letters(3)], "white"
    if kind == "yellow":
        return region, ["H", digits(6)], "#f2c500"
    if kind == "green":
        return region, ["M", digits(6)], "#1f9e4a"
    # personalized: a name, or random letters (so the model doesn't just memorize names)
    word = random.choice(NAMES) if random.random() < 0.5 else letters(random.randint(3, 7))
    return region, [word], "white"


def text_box(text, path, w, h, bg, ink):
    """Draw text, then squeeze it into exactly w x h -> tall, narrow letters like real plates."""
    font = ImageFont.truetype(path, 100)
    x0, y0, x1, y1 = font.getbbox(text)
    tmp = Image.new("RGB", (x1 - x0 + 4, y1 - y0 + 4), bg)
    ImageDraw.Draw(tmp).text((2 - x0, 2 - y0), text, font=font, fill=ink)
    return tmp.resize((w, h), Image.BILINEAR)


def draw_plate(region, parts, bg):
    W, H = 520, 112
    ink = "white" if bg == "#1f9e4a" else "black"
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, W - 3, H - 3], outline=ink, width=4)
    split = random.randint(70, 90)                                   # region box is narrow
    d.line([split, 4, split, H - 5], fill=ink, width=4)
    path = random.choice(FONTS)
    rh = random.randint(50, 70)                                      # region digits are SMALL
    img.paste(text_box(region, path, split - 20, rh, bg, ink), (10, (H - rh) // 2))
    text = random.choice(["", " "]).join(parts)                      # real plates: little or no gap
    tw = min(int(len(text) * random.uniform(40, 50)), 440 - split)   # long names get squeezed
    th = random.randint(78, 92)                                      # main letters are TALL
    img.paste(text_box(text, path, tw, th, bg, ink), (split + 8 + (440 - split - tw) // 2, (H - th) // 2))
    for i, color in enumerate(["#1eb5e6", "white", "#1eb53a"]):      # small flag + UZ
        d.rectangle([462, 22 + i * 10, 505, 31 + i * 10], fill=color, outline="black")
    d.text((468, 58), "UZ", font=ImageFont.truetype(path, 26), fill="#1e6fb5")
    return img


def put_on_car(plate):
    """Real crops show the car around the plate: plate fills only ~50-85% of the width."""
    pw, ph = plate.size
    W = int(pw / random.uniform(0.5, 0.85))
    H = int(W / random.uniform(1.5, 2.6))
    car = tuple(random.randint(20, 235) for _ in range(3))
    if random.random() < 0.5:                                         # many cars are white/gray
        car = (random.randint(150, 240),) * 3
    bgimg = Image.new("RGB", (W, H), car)
    d = ImageDraw.Draw(bgimg)
    for _ in range(random.randint(0, 3)):                             # bumper lines, grille, shadow
        y = random.randint(0, H)
        d.rectangle([0, y, W, y + random.randint(3, 25)], fill=tuple(random.randint(0, 255) for _ in range(3)))
    frame = random.randint(0, 12)                                     # dark plate holder
    holder = Image.new("RGB", (pw + 2 * frame, ph + 2 * frame), (random.randint(0, 50),) * 3)
    holder.paste(plate, (frame, frame))
    x = random.randint(0, max(0, W - holder.width))
    y = random.randint(int(H * 0.15), max(int(H * 0.15), H - holder.height - int(H * 0.1)))
    bgimg.paste(holder, (x, y))
    return bgimg.rotate(random.uniform(-6, 6), resample=Image.BILINEAR, fillcolor=car)


def make_split(name, n, seed):
    random.seed(seed)
    np.random.seed(seed)
    out = Path(OUT_DIR) / name
    (out / "images").mkdir(parents=True, exist_ok=True)
    with open(out / "annotations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "plate_text"])
        for i in range(n):
            region, parts, bg = random_plate()
            img = make_it_real(put_on_car(draw_plate(region, parts, bg)), hard=HARD).resize((256, 128))
            img.save(out / "images" / f"{i:06d}.png")
            w.writerow([f"images/{i:06d}.png", region + "".join(parts)])
            if (i + 1) % 5000 == 0:
                print(f"{name}: {i + 1}/{n}", flush=True)
    print(f"saved {n} plates -> {out}")


if __name__ == "__main__":
    n_train, n_val = int(sys.argv[1]), int(sys.argv[2])
    OUT_DIR = sys.argv[3] if len(sys.argv) > 3 else "data/ocr_synth"
    HARD = len(sys.argv) > 4 and sys.argv[4] == "hard"     # lesson 30: smaller plates + night IR
    make_split("train", n_train, seed=1)
    make_split("val", n_val, seed=2)
