"""Vehicle lesson 02: teach the VeRi model THIS place (fine-tuning), honestly.

Lesson 01: the VeRi model finds the right car first only ~46% of the time here: new country,
new cameras, new angles. Fine-tuning = keep training it a little on crops from these cameras.

The trap (and why the paper's 99.7% is not real): RoundaboutHD's train_images contain ALL 310
test cars, often from neighbouring frames. Training on them = seeing the exam answers.
Our honest split, by CAR (never by image):
  cars of the test set, sorted, every 2nd one  -> group A: their train_images are used for training
  the other test cars                          -> group B: never seen in training, used for the exam
  + the 201 train-only cars (seen by one camera only) are also used for training
Both the old and the new model are then scored on group B only, so the numbers compare fairly.

Training recipe (the usual "BoT" re-ID recipe):
  batch = 16 cars x 4 crops each
  loss  = cross-entropy "which car is this?" (label smoothing 0.1)
        + triplet: the hardest same-car crop must be closer than the hardest other-car crop
  augmentation: flip, small shift/crop, colour jitter, random erasing (fake occlusion)

Run:  uv run python vehicle/lessons/02_finetune.py            (train ~1 hour on an M4, then test)
      uv run python vehicle/lessons/02_finetune.py --test     (test the saved model only)
"""

import importlib
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "vehicle/lessons")
from fastreid_net import load_vehicle_reid  # noqa: E402

lesson01 = importlib.import_module("01_reid_test")

ROOT = lesson01.ROOT
DEVICE = lesson01.DEVICE
OUT = Path("vehicle/models/roundabout_finetuned.pt")
ITERATIONS = int(os.environ.get("ITERATIONS", 1000))
P, K, LR = 16, 4, 1e-4
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def split_cars():
    test_cars = sorted({lesson01.parse(f)[0] for f in (ROOT / "test_images").glob("*.jpg")})
    group_a, group_b = set(test_cars[0::2]), set(test_cars[1::2])
    return group_a, group_b


def augment(img):
    img = cv2.resize(img, (256, 256))
    if random.random() < 0.5:
        img = img[:, ::-1]
    pad = np.pad(img, ((10, 10), (10, 10), (0, 0)), mode="reflect")
    y, x = random.randint(0, 20), random.randint(0, 20)
    img = pad[y:y + 256, x:x + 256].astype(np.float32)
    img = np.clip(img * random.uniform(0.8, 1.2) + random.uniform(-20, 20), 0, 255)   # brightness / contrast
    if random.random() < 0.5:                                                            # random erasing
        h, w = random.randint(30, 120), random.randint(30, 120)
        y, x = random.randint(0, 256 - h), random.randint(0, 256 - w)
        img[y:y + h, x:x + w] = np.random.uniform(0, 255, (h, w, 3))
    return img


def batch_hard_triplet(feats, labels, margin=0.3):
    d = torch.cdist(feats, feats)
    same = labels[:, None] == labels[None, :]
    hardest_pos = (d * same).max(dim=1).values
    hardest_neg = (d + same * 1e6).min(dim=1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


def train():
    torch.set_grad_enabled(True)                     # lesson 01 switches gradients off (it only predicts)
    group_a, group_b = split_cars()
    by_car = defaultdict(list)
    for f in (ROOT / "train_images").glob("*.jpg"):
        car = lesson01.parse(f)[0]
        if car not in group_b:                       # group B cars are NEVER used for training
            by_car[car].append(f)
    cars = sorted(c for c, fs in by_car.items() if len(fs) >= K)
    label = {c: i for i, c in enumerate(cars)}
    print(f"training on {len(cars)} cars, {sum(len(by_car[c]) for c in cars)} crops; "
          f"exam on {len(group_b)} cars that training never sees")

    net = load_vehicle_reid("vehicle/models/veri_sbs_R50-ibn.pth").to(DEVICE).train()
    classifier = nn.Linear(2048, len(cars), bias=False).to(DEVICE)
    opt = torch.optim.Adam(list(net.parameters()) + list(classifier.parameters()), lr=LR, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERATIONS)
    t0 = time.perf_counter()
    for it in range(1, ITERATIONS + 1):
        chosen = random.sample(cars, P)
        files = [f for c in chosen for f in random.sample(by_car[c], K)]
        y = torch.tensor([label[lesson01.parse(f)[0]] for f in files], device=DEVICE)
        x = np.stack([augment(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)) for f in files])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(DEVICE)

        feat_map = net.backbone((x - net.pixel_mean) / net.pixel_std)
        pooled = F.adaptive_avg_pool2d(feat_map.clamp(min=1e-6).pow(net.gem_p), 1).pow(1.0 / net.gem_p)[..., 0, 0]
        neck = net.bottleneck(pooled[..., None, None])[..., 0, 0]
        logits = classifier(neck)
        loss_ce = F.cross_entropy(logits, y, label_smoothing=0.1)
        loss_tri = batch_hard_triplet(pooled, y)
        loss = loss_ce + loss_tri

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 100 == 0 or it == ITERATIONS:
            print(f"  iter {it}/{ITERATIONS}  ce {loss_ce.item():.3f}  triplet {loss_tri.item():.3f}  "
                  f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    net.eval()
    torch.save(net.state_dict(), OUT)
    print(f"saved {OUT}")


def test():
    _, group_b = split_cars()
    q_files, q_car, q_cam = lesson01.load_crops("query_images")
    g_files, g_car, g_cam = lesson01.load_crops("test_images")
    qm, gm = np.isin(q_car, list(group_b)), np.isin(g_car, list(group_b))

    cache = np.load(lesson01.CACHE)          # lesson 01 already embedded everything with the VeRi model
    rows = [("VeRi-SBS (no fine-tune)", cache["veri_q"][qm], cache["veri_g"][gm])]
    net = load_vehicle_reid("vehicle/models/veri_sbs_R50-ibn.pth")
    net.load_state_dict(torch.load(OUT, map_location="cpu", weights_only=True))
    net = net.to(DEVICE).eval()
    with torch.no_grad():
        rows.append(("VeRi-SBS + fine-tuned", lesson01.embed_veri([q_files[i] for i in np.where(qm)[0]], net),
                     lesson01.embed_veri([g_files[i] for i in np.where(gm)[0]], net)))

    print(f"\nexam: group B = {len(group_b)} cars never used for training, "
          f"{qm.sum()} queries, {gm.sum()} gallery crops")
    print(f"{'model':<24} {'Rank-1':>7} {'Rank-5':>7} {'mAP':>7}")
    for name, q, g in rows:
        r1, r5, m, _ = lesson01.evaluate(q, g, q_car[qm], q_cam[qm], g_car[gm], g_cam[gm])
        print(f"{name:<24} {r1:>7.1%} {r5:>7.1%} {m:>7.1%}")


if __name__ == "__main__":
    if "--test" not in sys.argv:
        train()
    test()
