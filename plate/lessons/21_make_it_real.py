"""OCR lesson 4: "break" clean fake plates so they look like street-camera crops.

Each step copies one problem the real camera has.
Run:  uv run python plate/lessons/21_make_it_real.py
"""

import importlib
import random
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, "plate/lessons")
fake = importlib.import_module("20_fake_plate")


def tilt(img):
    """Camera looks from the side -> the plate is a slanted 4-corner shape."""
    w, h = img.size
    dx, dy = w * random.uniform(-0.08, 0.08), h * random.uniform(-0.15, 0.15)
    # where each output corner takes its pixels from (top-left, bottom-left, bottom-right, top-right)
    quad = (dx, dy, -dx, h - dy, w + dx, h + dy, w - dx, -dy)
    return img.transform((w, h), Image.QUAD, quad, fill=1, fillcolor=(random.randint(40, 200),) * 3)


def far_away(img):
    """Car is far -> plate only ~60-200 px wide. Shrink, then grow back (detail is lost)."""
    w, h = img.size
    small_w = random.randint(60, 200)
    small = img.resize((small_w, max(12, small_w * h // w)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def light(img):
    """Night, sun, shadow -> darker/brighter, less contrast."""
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.35, 1.3))
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.2))


def blur(img):
    """Moving car or bad focus."""
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0, 2.5)))


def noise(img):
    """Cheap camera sensor at night -> grainy dots."""
    a = np.asarray(img).astype(np.float32)
    a += np.random.normal(0, random.uniform(0, 18), a.shape)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def very_far(img):
    """Lesson 30 showed most errors on 80-120 px plates: region "01" is ~10 px there.
    Crop is ~1.3x the plate width -> shrink the crop to 50-160 px."""
    w, h = img.size
    small_w = int(random.triangular(50, 160, 70))       # most often near 70 px
    small = img.resize((small_w, max(10, small_w * h // w)), Image.BILINEAR)
    return small.resize((w, h), random.choice([Image.BILINEAR, Image.NEAREST]))


def night_ir(img):
    """Infrared night camera: no colour, white plate glows, rest is dark."""
    g = np.asarray(img.convert("L")).astype(np.float32) / 255
    g = g ** random.uniform(1.5, 3.0)                    # darks get darker, the white plate stays bright
    g = g * random.uniform(0.8, 1.3) + random.uniform(0, 0.08)
    img = Image.fromarray((g.clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))   # glow


def make_it_real(img, hard=False):
    if hard and random.random() < 0.25:
        img = night_ir(img)
    steps = (tilt, light, very_far if hard and random.random() < 0.6 else far_away, blur, noise)
    for step in steps:
        if random.random() < 0.8:          # not every problem every time
            img = step(img)
    return img


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    sheet = Image.new("RGB", (520 * 2, 112 * 6), "gray")
    for i in range(6):
        clean = fake.draw_plate(*fake.random_text())
        sheet.paste(clean, (0, 112 * i))                    # left: clean
        sheet.paste(make_it_real(clean), (520, 112 * i))    # right: "real"
    sheet.save("plate/output/21_clean_vs_real.jpg")
    print("saved plate/output/21_clean_vs_real.jpg")
