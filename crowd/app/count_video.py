"""Crowd app: count the people in a video, a camera stream or a webcam, over time.

Uses our P2PNet fine-tuned on Tashkent metro and concert frames (crowd/models/p2pnet_tashkent.pt) and
counts at two image sizes (count_heads), so both far-away heads and big near faces are found.
Expect about 4-5 heads of error per frame on footage it has never seen (crowd/models/README.md).

Every --every seconds it counts one frame and writes
  crowd/output/count_<name>.mp4   the counted frames: a red dot on every head, the count, and a graph
                                  of the count so far (one video frame per counted moment)
  crowd/output/count_<name>.csv   second, count (and count inside --zone)
and prints the lowest, average and highest count at the end.

Run:  uv run python crowd/app/count_video.py VIDEO                       (file)
      uv run python crowd/app/count_video.py rtsp://user:pass@ip/stream  (camera stream)
      uv run python crowd/app/count_video.py 0                           (webcam)
options:
      --every 2                count one frame every 2 seconds (default 1)
      --start 60 --seconds 300 only that part of a file / stop a stream after 300 s
      --zone 0 500 1920 1080   also count only inside this box (x0 y0 x1 y1, pixels of the video),
                               e.g. a platform or the area in front of a stage
      --fps 4                  frames per second of the output video
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "crowd/lessons")
from p2pnet_net import count_heads, load_p2pnet  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "crowd/models/p2pnet_tashkent.pt"
OUT = Path("crowd/output")
MAX_SIDE = 1920        # the model was fine-tuned on 1080p frames; bigger ones are shrunk first
GRAPH_H = 120          # height of the count graph at the bottom of each output frame


def open_source(src):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        sys.exit(f"cannot open {src}")
    is_file = Path(src).expanduser().exists()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps > 120:         # streams often report 0, NaN or nonsense
        fps = 25.0
    return cap, fps, is_file


def frames(cap, fps, is_file, every, start, seconds):
    """Yields (second, frame) every `every` seconds. Files: frame numbers; streams/webcams: clock time."""
    if is_file:
        step = max(1, round(every * fps))
        n = int(start * fps)
        if n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        end = (start + seconds) * fps if seconds else float("inf")
        while n < end:
            ok, frame = cap.read()
            if not ok:
                return
            yield n / fps, frame
            for _ in range(step - 1):                  # grab() skips decoding to pixels: much faster than seeking
                if not cap.grab():
                    return
            n += step
    else:
        t0, last = time.time(), -every
        while True:
            ok, frame = cap.read()                     # keep reading so the stream buffer does not lag behind
            if not ok:
                return
            t = time.time() - t0
            if seconds and t > seconds:
                return
            if t - last >= every:
                last = t
                yield t, frame


def draw(frame, points, count, second, zone, zone_count, history):
    img = frame.copy()
    r = max(3, img.shape[1] // 400)
    for x, y in points.astype(int):
        cv2.circle(img, (x, y), r + 1, (0, 0, 0), -1)
        cv2.circle(img, (x, y), r, (0, 0, 255), -1)
    text = f"{count} people   {int(second // 60)}:{second % 60:04.1f}"
    if zone:
        cv2.rectangle(img, tuple(zone[:2]), tuple(zone[2:]), (0, 255, 255), 2)
        text += f"   in zone: {zone_count}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 56), (0, 0, 0), -1)
    cv2.putText(img, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    # graph of the count so far (the zone count if there is a zone)
    g = np.zeros((GRAPH_H, img.shape[1], 3), np.uint8)
    top = max(max(history), 1)
    if len(history) > 1:
        xs = np.linspace(10, img.shape[1] - 10, len(history))
        ys = GRAPH_H - 12 - (np.array(history) / top) * (GRAPH_H - 30)
        cv2.polylines(g, [np.c_[xs, ys].astype(np.int32)], False, (0, 200, 255), 2)
    cv2.putText(g, f"max {top}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return np.vstack([img, g])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="video file, stream URL, or webcam number")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between counted frames")
    ap.add_argument("--start", type=float, default=0.0, help="start at this second (files only)")
    ap.add_argument("--seconds", type=float, default=0.0, help="how many seconds to count (0 = all)")
    ap.add_argument("--zone", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--fps", type=float, default=4.0, help="frames per second of the output video")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    if not Path(args.model).exists():
        sys.exit(f"no model at {args.model}: run `git lfs pull` (it is stored with Git LFS)")

    net = load_p2pnet(args.model, DEVICE)
    cap, fps, is_file = open_source(args.source)
    name = Path(args.source).stem[:40] if is_file else "stream"
    OUT.mkdir(parents=True, exist_ok=True)
    writer, rows, history = None, [], []
    print(f"counting {args.source} every {args.every:g} s on {DEVICE}  (Ctrl+C stops and saves)")
    try:
        for second, frame in frames(cap, fps, is_file, args.every, args.start, args.seconds):
            s = min(1.0, MAX_SIDE / max(frame.shape[:2]))
            if s < 1:
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            points = count_heads(net, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            zone_count = None
            if args.zone:
                x0, y0, x1, y1 = args.zone
                zone_count = int(((points[:, 0] >= x0) & (points[:, 0] < x1) &
                                  (points[:, 1] >= y0) & (points[:, 1] < y1)).sum())
            rows.append((round(second, 1), len(points)) + ((zone_count,) if args.zone else ()))
            history.append(zone_count if args.zone else len(points))
            out = draw(frame, points, len(points), second, args.zone, zone_count, history)
            if writer is None:
                writer = cv2.VideoWriter(str(OUT / f"count_{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (out.shape[1], out.shape[0]))
            writer.write(out)
            print(f"\r{second:8.1f} s  {len(points):5d} people" + (f"  {zone_count:5d} in zone" if args.zone else ""),
                  end="", flush=True)
    except KeyboardInterrupt:
        pass
    cap.release()
    if writer:
        writer.release()
    if not rows:
        sys.exit("\nno frames read")
    with open(OUT / f"count_{name}.csv", "w", newline="") as f:
        csv.writer(f).writerows([("second", "count") + (("zone_count",) if args.zone else ()), *rows])
    c = np.array(history)
    peak = rows[int(c.argmax())][0]
    print(f"\n{len(rows)} frames counted{' (zone)' if args.zone else ''}: lowest {c.min()}, average {c.mean():.0f}, "
          f"highest {c.max()} at {int(peak // 60)}:{peak % 60:04.1f}")
    print(f"video: {OUT / f'count_{name}.mp4'}\ntable: {OUT / f'count_{name}.csv'}")


if __name__ == "__main__":
    main()
