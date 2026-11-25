"""Crowd lesson 03: fine-tune P2PNet on OUR labelled frames (from crowd/app/label_heads.py).

Starts from the authors' ShanghaiTech-A weights and keeps their recipe (models/p2pnet.py, train.py):
  each image -> 4 random 128x128 crops, random scale, random left-right flip
  Hungarian matching: every labelled head is paired with ONE predicted point
     (cost = 0.05 * distance - head score), all other predicted points should say "no head"
  loss = cross-entropy(head / no head; "no head" weighted 0.5) + 0.0002 * squared distance of the pairs
  AdamW, lr 1e-4 (heads) and 1e-5 (VGG backbone), gradient clipped at 0.1
One change for our cameras: the random scale range is wider (0.5-1.5 instead of 0.7-1.3), because
one frame of a long hall has heads from ~5 to ~200 pixels.

Test frames are whole videos left out of training (--test), so we measure a NEW place, not
another frame of a place the model already saw. Test frames must be labelled by hand, not --prefill.

Run:  uv run python crowd/lessons/03_finetune_p2pnet.py --test <video stem> [<video stem> ...]
Output: crowd/runs/p2pnet_ft/best.pt (lowest test MAE), log in the same folder.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "crowd/lessons")
from p2pnet_net import MEAN, STD, find_heads, load_p2pnet  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CROP, CROPS_PER_IMAGE = 128, 4
SCALE = (0.5, 1.5)
COST_POINT, COST_CLASS, EOS, POINT_LOSS = 0.05, 1.0, 0.5, 2e-4


def load_labels(root):
    """{video stem: [(image path, points (N, 2)), ...]}"""
    out = {}
    for js in sorted(Path(root).glob("*/*.json")):
        d = json.loads(js.read_text())
        out.setdefault(js.parent.name, []).append(
            (js.with_name(d["image"]), np.array(d["points"], np.float32).reshape(-1, 2), d.get("prefilled", False)))
    return out


def random_crops(img, pts):
    """Random scale + flip, then 4 random 128x128 crops: (4, 3, 128, 128) normalised, list of point arrays."""
    s = random.uniform(*SCALE)
    if min(img.shape[:2]) * s < CROP:
        s = CROP / min(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    pts = pts * s
    if random.random() < 0.5:
        img, pts = img[:, ::-1], np.c_[img.shape[1] - pts[:, 0], pts[:, 1]] if len(pts) else pts
    img = (img.astype(np.float32) / 255.0 - MEAN) / STD
    h, w = img.shape[:2]
    crops, targets = [], []
    for _ in range(CROPS_PER_IMAGE):
        y, x = random.randint(0, h - CROP), random.randint(0, w - CROP)
        crops.append(img[y:y + CROP, x:x + CROP])
        inside = (pts[:, 0] >= x) & (pts[:, 0] < x + CROP) & (pts[:, 1] >= y) & (pts[:, 1] < y + CROP)
        targets.append((pts[inside] - [x, y]).astype(np.float32))   # MPS has no float64
    return np.stack(crops), targets


def loss_fn(points, logits, targets):
    """P2PNet's loss: one-to-one matching, then classification + point distance."""
    ce, dist, n = 0.0, 0.0, max(1, sum(len(t) for t in targets))
    for p, lg, t in zip(points, logits, targets):
        labels = torch.zeros(len(p), dtype=torch.long, device=p.device)
        if len(t):
            t = torch.from_numpy(t).to(p.device)
            with torch.no_grad():
                cost = COST_POINT * torch.cdist(p, t) - COST_CLASS * lg.softmax(-1)[:, 1:2]
                rows, cols = linear_sum_assignment(cost.cpu().numpy())
            rows, cols = torch.as_tensor(rows, device=p.device), torch.as_tensor(cols, device=p.device)
            labels[rows] = 1
            dist = dist + ((p[rows] - t[cols]) ** 2).sum()
        ce = ce + F.cross_entropy(lg, labels, weight=torch.tensor([EOS, 1.0], device=p.device))
    return ce / len(points) + POINT_LOSS * dist / n


def evaluate(net, frames):
    errs = []
    for img_path, pts, _ in frames:
        rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        errs.append((len(find_heads(net, rgb)[0]), len(pts)))
    e = np.array(errs, np.float32)
    return dict(mae=np.abs(e[:, 0] - e[:, 1]).mean(), mape=100 * (np.abs(e[:, 0] - e[:, 1]) / np.maximum(e[:, 1], 1)).mean(),
                bias=(e[:, 0] - e[:, 1]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="crowd/data/labels")
    ap.add_argument("--test", nargs="+", required=True, help="video stems kept out of training")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8, help="images per step (x4 crops)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--out", default="crowd/runs/p2pnet_ft")
    args = ap.parse_args()
    random.seed(0), np.random.seed(0), torch.manual_seed(0)

    videos = load_labels(args.labels)
    test_videos = [v for v in videos if any(v.startswith(t) for t in args.test)]   # a prefix such as the YouTube id is enough
    missing = [t for t in args.test if not any(v.startswith(t) for v in videos)]
    if missing:
        sys.exit(f"no labels for test video(s) {missing}; labelled videos: {sorted(videos)}")
    test = [f for v in test_videos for f in videos[v]]
    train = [f for v, fs in videos.items() if v not in test_videos for f in fs]
    if any(f[2] for f in test):
        print("WARNING: some test frames were --prefill'ed from P2PNet; the test score will be too kind.")
    print(f"train: {len(train)} frames, {sum(len(f[1]) for f in train)} heads | "
          f"test: {len(test)} frames, {sum(len(f[1]) for f in test)} heads ({', '.join(test_videos)})")
    if not train:
        sys.exit("nothing to train on")
    images = [cv2.cvtColor(cv2.imread(str(f[0])), cv2.COLOR_BGR2RGB) for f in train]

    net = load_p2pnet("crowd/models/p2pnet_shtech_a.pth", DEVICE)
    best = evaluate(net.eval(), test)
    print(f"before fine-tuning  test MAE {best['mae']:.1f}  MAPE {best['mape']:.1f}%  bias {best['bias']:+.1f}")
    backbone = [p for n, p in net.named_parameters() if n.startswith("backbone")]
    rest = [p for n, p in net.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": rest, "lr": 1e-4}, {"params": backbone, "lr": 1e-5}], weight_decay=1e-4)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "best.pt")
    log = open(out / "log.txt", "a")
    steps = max(1, len(images) // args.batch)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train()
        for m in net.backbone.modules():          # keep BatchNorm statistics from ShanghaiTech: our batches are tiny
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        order, total = np.random.permutation(len(images)), 0.0
        for s in range(steps):
            crops, targets = [], []
            for i in order[s * args.batch:(s + 1) * args.batch]:
                c, t = random_crops(images[i], train[i][1])
                crops.append(c)
                targets += t
            x = torch.from_numpy(np.concatenate(crops)).permute(0, 3, 1, 2).to(DEVICE)
            points, logits = net(x)
            loss = loss_fn(points, logits, targets)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            opt.step()
            total += loss.item()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            r = evaluate(net.eval(), test)
            mark = ""
            if r["mae"] < best["mae"]:
                best, mark = r, "  <- best, saved"
                torch.save(net.state_dict(), out / "best.pt")
            line = (f"epoch {epoch:4d}  loss {total / steps:.4f}  test MAE {r['mae']:.1f}  MAPE {r['mape']:.1f}%"
                    f"  bias {r['bias']:+.1f}  ({time.time() - t0:.0f}s){mark}")
            print(line)
            log.write(line + "\n")
            log.flush()
    print(f"best test MAE {best['mae']:.1f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
