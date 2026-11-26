"""Vehicle lesson 04: add the Uzbek cars mined in lesson 03 to the training, test on BOTH places.

Start: the lesson 02 model (VeRi + RoundaboutHD). Keep training a little on a MIX:
  RoundaboutHD training cars (same as lesson 02: group B never used)
  + 2/3 of the Uzbek cars (split by CAR; the other 1/3 is the Uzbek exam, never seen)
Every batch: 12 RoundaboutHD cars + 4 Uzbek cars, 4 crops each (Uzbek cars are few, so they are
over-sampled; RoundaboutHD stays in the mix so the model does not forget it).

Two exams, three models (VeRi only / + RoundaboutHD / + RoundaboutHD + Uzbek):
  RoundaboutHD group B   same as lesson 02 (find the car in ANOTHER camera)
  Uzbek exam cars        one video = one camera, so instead of "another camera" we demand
                         "another moment": crops of the same car within 10 s of the query are removed,
                         the model must find the car from another angle / later in the video.
                         Plates are covered (lesson 03), so the model cannot just read them.

Run:  uv run python vehicle/lessons/04_finetune_uzbek.py          (train ~25 min, then both exams)
      uv run python vehicle/lessons/04_finetune_uzbek.py --test   (exams only)
Env:  ITERATIONS=500
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
lesson02 = importlib.import_module("02_finetune")

UZ = Path("vehicle/data/mined/reid")
OUT = Path("vehicle/models/roundabout_uzbek_finetuned.pt")
ITERATIONS = int(os.environ.get("ITERATIONS", 500))
P_RB, P_UZ, K, LR = 12, 4, 4, 5e-5
GAP_FRAMES = 10 * 30          # "another moment" = more than ~10 s away (videos are 25-30 fps)
DEVICE = lesson01.DEVICE
random.seed(0)
torch.manual_seed(0)


def uz_files():
    files = sorted(UZ.glob("*.jpg"))
    car = np.array([f.stem.split("_")[0] for f in files])
    video = np.array([f.stem.split("_")[1] for f in files])
    frame = np.array([int(f.stem.split("_")[2]) // 100 for f in files])   # lesson 03 glued the vehicle number on
    return files, car, video, frame


EXAM_LIST = UZ.parent / "uz_exam_cars.txt"


def uz_split():
    """train 2/3, exam 1/3 of the cars. The exam was frozen into EXAM_LIST when more videos were added,
    so new cars only ever go to training and all results stay comparable."""
    cars = sorted(set(uz_files()[1]))
    exam = set(EXAM_LIST.read_text().split()) if EXAM_LIST.exists() else set(cars[0::3])
    return set(cars) - exam, exam


def load_net(weights):
    net = load_vehicle_reid("vehicle/models/veri_sbs_R50-ibn.pth")
    if weights:
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return net.to(DEVICE)


def train():
    torch.set_grad_enabled(True)
    _, group_b = lesson02.split_cars()
    rb = defaultdict(list)
    for f in (lesson01.ROOT / "train_images").glob("*.jpg"):
        if lesson01.parse(f)[0] not in group_b:
            rb[lesson01.parse(f)[0]].append(f)
    uz_train, _ = uz_split()
    files, car, _, _ = uz_files()
    uz = defaultdict(list)
    for f, c in zip(files, car):
        if c in uz_train:
            uz["uz" + c].append(f)
    rb_cars = sorted(c for c, fs in rb.items() if len(fs) >= K)
    uz_cars = sorted(c for c, fs in uz.items() if len(fs) >= K)
    pool = {**rb, **uz}
    label = {c: i for i, c in enumerate(rb_cars + uz_cars)}
    print(f"training: {len(rb_cars)} RoundaboutHD cars + {len(uz_cars)} Uzbek cars "
          f"({sum(len(uz[c]) for c in uz_cars)} crops)")

    net = load_net(lesson02.OUT).train()
    classifier = nn.Linear(2048, len(label), bias=False).to(DEVICE)
    opt = torch.optim.Adam(list(net.parameters()) + list(classifier.parameters()), lr=LR, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERATIONS)
    t0 = time.perf_counter()
    for it in range(1, ITERATIONS + 1):
        chosen = random.sample(rb_cars, P_RB) + random.sample(uz_cars, P_UZ)
        picked = [(c, f) for c in chosen for f in random.sample(pool[c], K)]
        y = torch.tensor([label[c] for c, _ in picked], device=DEVICE)
        x = np.stack([lesson02.augment(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)) for _, f in picked])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(DEVICE)

        feat_map = net.backbone((x - net.pixel_mean) / net.pixel_std)
        pooled = F.adaptive_avg_pool2d(feat_map.clamp(min=1e-6).pow(net.gem_p), 1).pow(1.0 / net.gem_p)[..., 0, 0]
        neck = net.bottleneck(pooled[..., None, None])[..., 0, 0]
        loss_ce = F.cross_entropy(classifier(neck), y, label_smoothing=0.1)
        loss_tri = lesson02.batch_hard_triplet(pooled, y)
        loss = loss_ce + loss_tri
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 50 == 0 or it == ITERATIONS:
            print(f"  iter {it}/{ITERATIONS}  ce {loss_ce.item():.3f}  triplet {loss_tri.item():.3f}  "
                  f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    torch.save(net.eval().state_dict(), OUT)
    print(f"saved {OUT}")


def evaluate_uz(v, car, video, frame):
    """Rank-1 / Rank-5 / mAP; for each query, same-car crops within GAP_FRAMES (same video) are removed"""
    sims = v @ v.T
    r1 = r5 = 0
    aps = []
    for i in range(len(v)):
        near = (car == car[i]) & (video == video[i]) & (np.abs(frame - frame[i]) < GAP_FRAMES)
        keep = ~near                                    # also removes the query itself
        order = np.argsort(-sims[i][keep])
        hits = car[keep][order] == car[i]
        if not hits.any():
            continue
        r1 += hits[0]
        r5 += hits[:5].any()
        ranks = np.where(hits)[0] + 1
        aps.append(np.mean(np.arange(1, len(ranks) + 1) / ranks))
    n = len(aps)
    return r1 / n, r5 / n, float(np.mean(aps)), n


def test():
    torch.set_grad_enabled(False)
    _, group_b = lesson02.split_cars()
    q_files, q_car, q_cam = lesson01.load_crops("query_images")
    g_files, g_car, g_cam = lesson01.load_crops("test_images")
    qm, gm = np.isin(q_car, list(group_b)), np.isin(g_car, list(group_b))
    _, uz_exam = uz_split()
    files, car, video, frame = uz_files()
    um = np.isin(car, list(uz_exam))

    models = [("VeRi only", None), ("+ RoundaboutHD (lesson 02)", lesson02.OUT), ("+ RoundaboutHD + Uzbek", OUT)]
    print(f"\nRoundaboutHD exam: {len(group_b)} cars, {qm.sum()} queries | "
          f"Uzbek exam: {len(uz_exam)} cars, {um.sum()} crops (never trained on)")
    print(f"{'model':<28} {'RB R-1':>7} {'RB mAP':>7}   {'UZ R-1':>7} {'UZ R-5':>7} {'UZ mAP':>7}")
    cache = np.load(lesson01.CACHE)
    for name, weights in models:
        if weights is not None and not Path(weights).exists():
            continue
        net = load_net(weights).eval()
        if weights is None:
            q, g = cache["veri_q"][qm], cache["veri_g"][gm]
        else:
            q = lesson01.embed_veri([q_files[i] for i in np.where(qm)[0]], net)
            g = lesson01.embed_veri([g_files[i] for i in np.where(gm)[0]], net)
        rb = lesson01.evaluate(q, g, q_car[qm], q_cam[qm], g_car[gm], g_cam[gm])
        u = lesson01.embed_veri([files[i] for i in np.where(um)[0]], net)
        uz = evaluate_uz(u, car[um], video[um], frame[um])
        print(f"{name:<28} {rb[0]:>7.1%} {rb[2]:>7.1%}   {uz[0]:>7.1%} {uz[1]:>7.1%} {uz[2]:>7.1%}", flush=True)
    print(f"(Uzbek queries that have the same car > 10 s away: {uz[3]})")


if __name__ == "__main__":
    if "--test" not in sys.argv:
        train()
    test()
