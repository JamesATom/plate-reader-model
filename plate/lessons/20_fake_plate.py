"""OCR lesson 3: draw a fake Uzbek plate. The text is known, so no hand labelling.

Plate layout (white, private car):  | 01 | A 123 BC  [flag UZ] |
Run:  uv run python plate/lessons/20_fake_plate.py
"""

import random

from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVXYZ"
DIGITS = "0123456789"


def random_text():
    region = random.choice(["01", "10", "20", "25", "30", "40", "50", "60", "70", "75", "80", "85", "90", "95"])
    rest = random.choice(LETTERS) + "".join(random.choices(DIGITS, k=3)) + "".join(random.choices(LETTERS, k=2))
    return region, rest


def draw_plate(region, rest):
    W, H = 520, 112                                  # real plate is 520 x 112 mm
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, W - 3, H - 3], outline="black", width=4)     # black border
    d.line([100, 4, 100, H - 5], fill="black", width=4)             # line after region
    font = ImageFont.truetype(FONT, 88)
    d.text((14, 10), region, font=font, fill="black")
    d.text((115, 10), rest[0] + " " + rest[1:4] + " " + rest[4:], font=font, fill="black")
    # flag: blue / white / green stripes, then "UZ"
    for i, color in enumerate(["#1eb5e6", "white", "#1eb53a"]):
        d.rectangle([455, 14 + i * 14, 505, 27 + i * 14], fill=color, outline="black")
    d.text((462, 62), "UZ", font=ImageFont.truetype(FONT, 34), fill="#1e6fb5")
    return img


if __name__ == "__main__":
    random.seed(0)
    sheet = Image.new("RGB", (520, 112 * 6), "gray")
    for i in range(6):
        region, rest = random_text()
        sheet.paste(draw_plate(region, rest), (0, 112 * i))
        print(region + rest)
    sheet.save("plate/output/20_fake_plates.jpg")
    print("saved plate/output/20_fake_plates.jpg")
