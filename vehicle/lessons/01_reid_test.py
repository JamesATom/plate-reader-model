"""Vehicle lesson 01: how well does a re-ID model find the same car in ANOTHER camera?

Data: RoundaboutHD ReID subset (University of Bath, MIT licence), 4 real 4K street cameras.
  query_images/  1,171 crops: "find this car"
  test_images/  23,225 crops: the gallery to search in
  name = {car id}_{camera}_{frame}.jpg

Standard re-ID protocol (same as VeRi / Market-1501):
  for each query, gallery crops of the SAME car from the SAME camera are removed,
  so the model must find the car in a DIFFERENT camera.
  Rank-1  = the closest gallery crop is the right car
  Rank-5  = the right car is among the 5 closest
  mAP     = how high ALL crops of the right car are ranked (1.0 = all of them first)

Models:
  VeRi-SBS  FastReID SBS R50-ibn trained on VeRi-776 (other country, other cameras: never saw this place)
  DINOv2    general-purpose image model, never trained for re-ID (a "no special training" baseline)

Run:  uv run python vehicle/lessons/01_reid_test.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "vehicle/lessons")
from fastreid_net import load_vehicle_reid  # noqa: E402

ROOT = Path("vehicle/data/RoundaboutHD/RoundaboutHD_Reid_subset")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CACHE = Path("vehicle/output/v01_embeddings.npz")
torch.set_grad_enabled(False)


def parse(path):
    car, cam, frame = path.stem.split("_")
    return car, cam, int(frame)


def load_crops(folder):
    files = sorted((ROOT / folder).glob("*.jpg"))
    return files, np.array([parse(f)[0] for f in files]), np.array([parse(f)[1] for f in files])


def embed_veri(files, net, batch=64):
    out = []
    for i in range(0, len(files), batch):
        imgs = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files[i:i + batch]]
        x = torch.from_numpy(np.stack([cv2.resize(im, (256, 256)) for im in imgs]).astype(np.float32))
        out.append(net(x.permute(0, 3, 1, 2).to(DEVICE)).cpu().numpy())
    v = np.concatenate(out)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def embed_dino(files, batch=64):
    from transformers import AutoModel
    net = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    out = []
    for i in range(0, len(files), batch):
        imgs = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files[i:i + batch]]
        x = (np.stack([cv2.resize(im, (224, 224)) for im in imgs]) / 255.0 - mean) / std
        x = torch.from_numpy(x.astype(np.float32)).permute(0, 3, 1, 2).to(DEVICE)
        out.append(net(pixel_values=x).pooler_output.cpu().numpy())   # the [CLS] summary vector
    v = np.concatenate(out)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def evaluate(q, g, q_car, q_cam, g_car, g_cam):
    """Rank-1, Rank-5, mAP with the standard same-car-same-camera removal."""
    sims = q @ g.T
    r1 = r5 = 0
    aps = []
    for i in range(len(q)):
        keep = ~((g_car == q_car[i]) & (g_cam == q_cam[i]))
        order = np.argsort(-sims[i][keep])
        hits = (g_car[keep][order] == q_car[i])
        if not hits.any():
            continue
        r1 += hits[0]
        r5 += hits[:5].any()
        ranks = np.where(hits)[0] + 1
        aps.append(np.mean(np.arange(1, len(ranks) + 1) / ranks))
    n = len(aps)
    return r1 / n, r5 / n, float(np.mean(aps)), n


if __name__ == "__main__":
    q_files, q_car, q_cam = load_crops("query_images")
    g_files, g_car, g_cam = load_crops("test_images")
    print(f"query {len(q_files)} crops, gallery {len(g_files)} crops, {len(set(q_car))} cars, cameras {sorted(set(g_cam))}")

    if CACHE.exists():
        d = dict(np.load(CACHE))
    else:
        d = {}
        t0 = time.perf_counter()
        veri = load_vehicle_reid("vehicle/models/veri_sbs_R50-ibn.pth").to(DEVICE)
        d["veri_q"], d["veri_g"] = embed_veri(q_files, veri), embed_veri(g_files, veri)
        print(f"  VeRi-SBS embedded ({time.perf_counter() - t0:.0f}s)")
        d["dino_q"], d["dino_g"] = embed_dino(q_files), embed_dino(g_files)
        print(f"  DINOv2 embedded ({time.perf_counter() - t0:.0f}s)")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, **d)

    print(f"\n{'model':<10} {'Rank-1':>7} {'Rank-5':>7} {'mAP':>7}   queries")
    for name, key in [("VeRi-SBS", "veri"), ("DINOv2", "dino")]:
        r1, r5, m, n = evaluate(d[f"{key}_q"], d[f"{key}_g"], q_car, q_cam, g_car, g_cam)
        print(f"{name:<10} {r1:>7.1%} {r5:>7.1%} {m:>7.1%}   {n}")
    print("\npaper (FastReID SBS, ImageNet only, no vehicle training):  Rank-1 44.8%  mAP 19.5%")
