"""Crowd lesson 02: count people in frames of a real video and SEE what each model counted.

No true count here, so we judge by eye: each output image shows
  left    P2PNet-A head points (red dots), count in the corner
  middle  CLIP-EBC-A density map (bright = many people per block)
  right   CLIP-EBC-B density map
Things to look for: dots on things that are not heads (lamps, signs, logos), missed far-away heads,
big near heads counted twice or not at all.

Run:  uv run python crowd/lessons/02_count_video_frames.py VIDEO SECOND [SECOND ...]
e.g.  uv run python crowd/lessons/02_count_video_frames.py ~/Downloads/metro.mp4 6 8 20 24 36
Output: crowd/output/c02_<video>_<second>s.jpg
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "crowd/lessons")
from clip_ebc_net import density_map, load_clip_ebc  # noqa: E402
from p2pnet_net import find_heads, load_p2pnet  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = Path("crowd/output")
MAX_SIDE = 1920    # bigger frames are shrunk first (P2PNet was trained on photos ~1000 px wide)


def read_frame(video, second):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(second * cap.get(cv2.CAP_PROP_FPS))))
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"cannot read {video} at {second}s")
    scale = min(1.0, MAX_SIDE / max(frame.shape[:2]))
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(img, text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    return img


def draw_points(bgr, points, count):
    img = bgr.copy()
    for x, y in points.astype(int):
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
    return label(img, f"P2PNet-A {count}")


def draw_density(bgr, dens, name):
    heat = cv2.resize(dens, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(heat / max(dens.max(), 1e-6), 0, 1)), cv2.COLORMAP_JET)
    return label(cv2.addWeighted(bgr, 0.45, heat, 0.55, 0), f"{name} {dens.sum():.0f}")


video, seconds = Path(sys.argv[1]).expanduser(), [float(s) for s in sys.argv[2:]]
p2p = load_p2pnet("crowd/models/p2pnet_shtech_a.pth", DEVICE)
ebc_a = load_clip_ebc("crowd/models/clip_ebc_vitb16_shtech_a.pth", "sha", DEVICE)
ebc_b = load_clip_ebc("crowd/models/clip_ebc_vitb16_shtech_b.pth", "shb", DEVICE)
OUT.mkdir(parents=True, exist_ok=True)

print(f"{'second':>6} {'size':>10} {'P2PNet-A':>9} {'EBC-A':>7} {'EBC-B':>7}  seconds (P2P / A / B)")
for s in seconds:
    bgr = read_frame(video, s)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    points, _ = find_heads(p2p, rgb)
    t1 = time.perf_counter()
    da = density_map(ebc_a, rgb)
    t2 = time.perf_counter()
    db = density_map(ebc_b, rgb)
    t3 = time.perf_counter()
    print(f"{s:6.1f} {bgr.shape[1]:>4}x{bgr.shape[0]:<5} {len(points):9d} {da.sum():7.0f} {db.sum():7.0f}"
          f"  {t1 - t0:.2f} / {t2 - t1:.2f} / {t3 - t2:.2f}")
    panel = np.hstack([draw_points(bgr, points, len(points)),
                       draw_density(bgr, da, "CLIP-EBC-A"), draw_density(bgr, db, "CLIP-EBC-B")])
    cv2.imwrite(str(OUT / f"c02_{video.stem[:20]}_{s:g}s.jpg"), panel)
print(f"\nimages: {OUT}/c02_*.jpg")
