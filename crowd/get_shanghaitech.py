"""Get the ShanghaiTech TEST images and true counts without a Kaggle login.

Source: HuggingFace dataset KTAEHWA/shanghaitech-crowd-counting (a re-upload of ShanghaiTech, research use
only). It has each image and its true count, but NOT the head points, so it is good for testing
(lesson 01) but not for training. For training use the Kaggle copy, which has the .mat point files.

Writes crowd/data/ShanghaiTech/part_{A,B}/test_data/images/IMG_n.png and .../test_data/counts.csv
Checks: SHA-256 of the downloads; 182 + 316 images; mean count 433.3 (A) and 123.8 (B), as published.

Run:  uv run python crowd/get_shanghaitech.py
"""

import csv
import hashlib
import io
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

URL = "https://huggingface.co/datasets/KTAEHWA/shanghaitech-crowd-counting/resolve/main/data"
PARTS = {   # part: (parquet file, sha256, images, mean count)
    "part_A": ("part_A_test_data-00000-of-00001-c5e0a32674388d67.parquet",
               "6e5d049c797efef47734843878f095fe0f0992a539e2055b39bb17876ea5a137", 182, 433.3),
    "part_B": ("part_B_test_data-00000-of-00001-5fa64741044e9db4.parquet",
               "26888f9b4442eebc5aeeff70018850061d6c4c166e727e97ee591564b26abf61", 316, 123.8),
}
DOWNLOADS = Path("crowd/data/hf_shanghaitech")
OUT = Path("crowd/data/ShanghaiTech")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


for part, (name, expected, n, mean) in PARTS.items():
    src = DOWNLOADS / name
    if not (src.exists() and sha256(src) == expected):
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        print(f"{part}: downloading {name}")
        request = urllib.request.Request(f"{URL}/{name}", headers={"User-Agent": "crowd-data-download"})
        with urllib.request.urlopen(request) as response, open(src, "wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
        if sha256(src) != expected:
            src.unlink()
            sys.exit(f"{part}: checksum mismatch, file deleted.")

    rows = pq.read_table(src, columns=["file_name", "image", "count"]).to_pylist()
    counts = [r["count"] for r in rows]
    if len(rows) != n or abs(sum(counts) / n - mean) > 0.1:
        sys.exit(f"{part}: expected {n} images with mean {mean}, got {len(rows)} with mean {sum(counts) / len(rows):.1f}")

    folder = OUT / part / "test_data"
    (folder / "images").mkdir(parents=True, exist_ok=True)
    for r in rows:
        stem = Path(r["file_name"]).stem           # IMG_12
        path = folder / "images" / f"{stem}.png"
        if not path.exists():
            img = Image.open(io.BytesIO(r["image"]["bytes"]))
            img.load()                              # decode fully: a broken file fails here, not later
            path.write_bytes(r["image"]["bytes"])
    with open(folder / "counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "count"])
        w.writerows(sorted(((Path(r["file_name"]).stem, r["count"]) for r in rows),
                           key=lambda x: int(x[0].split("_")[1])))
    print(f"{part}: {n} images, mean count {sum(counts) / n:.1f} -> {folder}")
