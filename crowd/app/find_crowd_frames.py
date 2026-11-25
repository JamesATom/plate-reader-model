"""Crowd tool: scan a video and find the moments worth labelling.

News reports and fan videos are mostly presenters, stages and titles; crowds appear in short parts.
This runs P2PNet (fast) on one frame every --step seconds and
  - writes crowd/output/scan_<video>.csv   (second, P2PNet count)
  - writes crowd/output/scan_<video>.jpg   (thumbnails of the crowd moments, with second and count)
  - prints suggested --seconds for label_heads.py: frames spread over the whole range of crowd sizes,
    at least --gap seconds apart (neighbouring frames look almost the same, so they teach little).
The counts are only a guide for choosing frames: P2PNet undercounts our footage.

Run:  uv run python crowd/app/find_crowd_frames.py VIDEO [--step 2] [--min-count 15] [--pick 15]
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "crowd/lessons")
from p2pnet_net import find_heads, load_p2pnet  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = Path("crowd/output")
MAX_SIDE = 1920


def scan(video, step, net):
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows, thumbs = [], {}
    for sec in np.arange(0, n / fps, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ok, bgr = cap.read()
        if not ok:
            break
        s = min(1.0, MAX_SIDE / max(bgr.shape[:2]))
        small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
        count = len(find_heads(net, cv2.cvtColor(small, cv2.COLOR_BGR2RGB))[0])
        rows.append((round(float(sec), 1), count))
        thumbs[rows[-1][0]] = cv2.resize(bgr, (240, round(240 * bgr.shape[0] / bgr.shape[1])))
    return rows, thumbs


def pick(rows, k, gap, min_count):
    """Frames spread over the range of counts: aim at k evenly spaced count levels (lowest to highest),
    for each take the closest-count frame that is >= gap seconds from those already taken."""
    cand = [(s, c) for s, c in rows if c >= min_count]
    if not cand:
        return []
    targets = np.quantile([c for _, c in cand], np.linspace(1, 0, k))   # highest first: dense frames matter most
    chosen = []
    for t in targets:
        free = [(s, c) for s, c in cand if all(abs(s - u) >= gap for u, _ in chosen)]
        if not free:
            break
        chosen.append(min(free, key=lambda x: abs(x[1] - t)))
    return sorted(chosen)


def contact_sheet(rows, thumbs, min_count, path, cols=8):
    tiles = []
    for s, c in rows:
        if c >= min_count:
            t = thumbs[s].copy()
            cv2.rectangle(t, (0, 0), (t.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(t, f"{s:g}s  {c}", (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(t)
    if not tiles:
        return False
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), grid[:12000])          # keep the image a sane size for very long videos
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--min-count", type=int, default=15, help="frames with fewer heads are not 'crowd'")
    ap.add_argument("--pick", type=int, default=15)
    ap.add_argument("--gap", type=float, default=8.0)
    args = ap.parse_args()
    video = args.video.expanduser()
    net = load_p2pnet("crowd/models/p2pnet_shtech_a.pth", DEVICE)
    rows, thumbs = scan(video, args.step, net)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = video.stem[:40]
    with open(OUT / f"scan_{stem}.csv", "w", newline="") as f:
        csv.writer(f).writerows([("second", "p2pnet_count"), *rows])
    crowd = [r for r in rows if r[1] >= args.min_count]
    print(f"{video.name}: {len(rows)} frames scanned, {len(crowd)} with >= {args.min_count} heads"
          f" ({100 * len(crowd) / max(1, len(rows)):.0f}%), max {max((c for _, c in rows), default=0)}")
    if contact_sheet(rows, thumbs, args.min_count, OUT / f"scan_{stem}.jpg"):
        chosen = pick(rows, args.pick, args.gap, args.min_count)
        print(f"  sheet: {OUT / f'scan_{stem}.jpg'}")
        print(f"  suggested: --seconds {' '.join(f'{s:g}' for s, _ in chosen)}")
        print(f"  (P2PNet counts {', '.join(str(c) for _, c in chosen)})")


if __name__ == "__main__":
    main()
