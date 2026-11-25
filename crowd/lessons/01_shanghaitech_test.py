"""Crowd lesson 01: how close are the ready-made crowd counters to the true head count?

Data: ShanghaiTech (Zhang et al. 2016, research use only), the standard crowd counting benchmark.
  Part A: 182 test photos from the internet, dense crowds (33 to 3,139 people, mean ~430)
  Part B: 316 test photos from Shanghai street cameras (9 to 578 people, mean ~120)
  every person's head is marked with one point, so the true count is the number of points.
  Get it with  uv run python crowd/get_shanghaitech.py  (test images + counts, from HuggingFace)
  or from https://www.kaggle.com/datasets/tthien/shanghaitech (needs a Kaggle login, also has the head
  points) unzipped into crowd/data/, so that crowd/data/**/part_A/test_data/images/IMG_1.jpg exists.

Models (crowd/download_models.py):
  P2PNet-A     finds one point per head (VGG16), trained on Part A.   Authors: MAE 51.96 on Part A
  CLIP-EBC-A   CLIP ViT-B/16 sorts each 8x8 block into 0/1/2/3/4+ people, trained on Part A.
               Authors' log: MAE 53.2, RMSE 85.9 on Part A
  CLIP-EBC-B   same, trained on Part B. Authors' log: MAE 6.68, RMSE 10.5 on Part B
  If our numbers match the authors', our re-written network code is right.
  Every model is also run on the OTHER part: that shows what happens on scenes it was not trained on.

  MAE  = average |predicted - true| people per image
  RMSE = like MAE but big misses count more
  MAPE = average |predicted - true| / true (a 10-person miss means more in a 20-person scene)

Run:  uv run python crowd/lessons/01_shanghaitech_test.py
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.io
import torch

sys.path.insert(0, "crowd/lessons")
from clip_ebc_net import density_map, load_clip_ebc  # noqa: E402
from p2pnet_net import find_heads, load_p2pnet  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = Path("crowd/output/c01_counts.csv")
MODELS = {
    "P2PNet-A": lambda: load_p2pnet("crowd/models/p2pnet_shtech_a.pth", DEVICE),
    "CLIP-EBC-A": lambda: load_clip_ebc("crowd/models/clip_ebc_vitb16_shtech_a.pth", "sha", DEVICE),
    "CLIP-EBC-B": lambda: load_clip_ebc("crowd/models/clip_ebc_vitb16_shtech_b.pth", "shb", DEVICE),
}
BUCKETS = [(0, 50), (50, 150), (150, 500), (500, 1000), (1000, 10**6)]


def find_part(part):
    hits = sorted(Path("crowd/data").glob(f"**/{part}/test_data/images"))
    if not hits:
        sys.exit(f"ShanghaiTech {part} not found under crowd/data/. See the download note at the top of this file.")
    return hits[0].parent


def load_test_set(part):
    """[(image path, true count)]. Counts come from the .mat point files (Kaggle copy) or counts.csv (HF copy)."""
    root = find_part(part)
    counts = {}
    if (root / "counts.csv").exists():
        with open(root / "counts.csv") as f:
            counts = {r["image"]: int(r["count"]) for r in csv.DictReader(f)}
    items = []
    images = list(root.glob("images/*.jpg")) + list(root.glob("images/*.png"))
    for img in sorted(images, key=lambda p: int(p.stem.split("_")[1])):
        if img.stem in counts:
            items.append((img, counts[img.stem]))
        else:
            mat = scipy.io.loadmat(root / "ground-truth" / f"GT_{img.stem}.mat")
            items.append((img, len(mat["image_info"][0, 0][0, 0][0])))
    return items


def count(name, net, image_rgb):
    if name.startswith("P2PNet"):
        return float(len(find_heads(net, image_rgb)[0]))
    return float(density_map(net, image_rgb).sum())


def run_all():
    """Count every test image with every model. Cached in the CSV, so re-runs only print."""
    if OUT.exists():
        with open(OUT) as f:
            return list(csv.DictReader(f))
    rows = []
    sets = {part: load_test_set(part) for part in ("part_A", "part_B")}
    for name, load in MODELS.items():
        net = load()
        for part, items in sets.items():
            count(name, net, cv2.cvtColor(cv2.imread(str(items[0][0])), cv2.COLOR_BGR2RGB))   # warm-up
            for img_path, truth in items:
                image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
                if DEVICE == "mps":
                    torch.mps.synchronize()
                t = time.perf_counter()
                pred = count(name, net, image)
                rows.append({"model": name, "part": part, "image": img_path.name, "true": truth,
                             "pred": round(pred, 2), "seconds": round(time.perf_counter() - t, 4),
                             "width": image.shape[1], "height": image.shape[0]})
            print(f"  {name:<11} {part} done")
        del net
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return [{k: str(v) for k, v in r.items()} for r in rows]


def scores(rows):
    t = np.array([float(r["true"]) for r in rows])
    p = np.array([float(r["pred"]) for r in rows])
    return dict(n=len(t), mae=np.abs(p - t).mean(), rmse=np.sqrt(((p - t) ** 2).mean()),
                mape=100 * (np.abs(p - t) / t).mean(), bias=(p - t).mean(),
                sec=np.mean([float(r["seconds"]) for r in rows]))


rows = run_all()
for part, label in [("part_A", "Part A (dense, internet photos)"), ("part_B", "Part B (street cameras)")]:
    print(f"\n{label}")
    print(f"  {'model':<11} {'MAE':>7} {'RMSE':>7} {'MAPE':>6} {'bias':>7} {'s/img':>6}")
    for name in MODELS:
        s = scores([r for r in rows if r["model"] == name and r["part"] == part])
        print(f"  {name:<11} {s['mae']:7.1f} {s['rmse']:7.1f} {s['mape']:5.1f}% {s['bias']:+7.1f} {s['sec']:6.2f}")

print("\nBy true crowd size (Part A + Part B together): MAE / MAPE")
print(f"  {'people':<10} {'images':>6}" + "".join(f" {n:>17}" for n in MODELS))
for lo, hi in BUCKETS:
    line = ""
    for name in MODELS:
        sel = [r for r in rows if r["model"] == name and lo <= float(r["true"]) < hi]
        if not sel:
            break
        s = scores(sel)
        line += f" {s['mae']:8.1f} / {s['mape']:4.0f}%"
    if line:
        print(f"  {f'{lo}-{hi}' if hi < 10**6 else f'{lo}+':<10} {len(sel):>6}{line}")
print(f"\nPer-image counts: {OUT}  (bias > 0 = counts too many)")
