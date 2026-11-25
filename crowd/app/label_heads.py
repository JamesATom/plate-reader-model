"""Crowd tool: click every head in video frames. The points become our true counts (to measure the
models) and later our training data (to fine-tune P2PNet).

A big frame has hundreds of small heads, so it is shown in TILES, each one enlarged to fill the window.
The dim border around a tile is the neighbouring area: heads you already clicked there are shown,
so heads on a tile edge are not clicked twice.

Run:  uv run python crowd/app/label_heads.py VIDEO --every 5            (a frame every 5 seconds)
      uv run python crowd/app/label_heads.py VIDEO --seconds 7 20 36    (chosen moments)
      add --prefill to start from P2PNet's points and only correct them (much faster, but use it
      only for TRAINING frames: for test frames it would push the labels towards P2PNet's answer)

Mouse:  left click            add a head (click the middle of the head)
        right click or shift+click   remove the nearest head
Keys:   n / space   next tile          b   previous tile
        o           whole-frame overview (toggle); clicks work there too
        u           undo last change
        q           save and go to the next frame      Esc   save and stop

Saved after every change: crowd/data/labels/<video>/<video>_<second>s.jpg (full-size frame)
                          and .json {"image", "video", "second", "points": [[x, y], ...]}
Re-running continues where you left off.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

WIN = "label heads"
VIEW_W, VIEW_H = 1400, 880         # window size; tiles are enlarged to fit it
TILE = 480                         # tile size in frame pixels
LABELS = Path("crowd/data/labels")


def grab(video, second):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(second * cap.get(cv2.CAP_PROP_FPS))))
    ok, frame = cap.read()
    return frame if ok else None


def video_seconds(video):
    cap = cv2.VideoCapture(str(video))
    return cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1e-6)


class Labeler:
    def __init__(self, img, points, save):
        self.img, self.points, self.save = img, [tuple(p) for p in points], save
        self.history = []                      # ("add", point) or ("remove", point)
        h, w = img.shape[:2]
        rows, cols = math.ceil(h / TILE), math.ceil(w / TILE)
        self.tiles = [(r * h // rows, (r + 1) * h // rows, c * w // cols, (c + 1) * w // cols)
                      for r in range(rows) for c in range(cols)]
        self.i, self.overview = 0, False

    def view(self):
        """The part of the frame on screen (y0, y1, x0, x1) and its enlargement factor."""
        h, w = self.img.shape[:2]
        if self.overview:
            box = (0, h, 0, w)
        else:
            y0, y1, x0, x1 = self.tiles[self.i]
            m = TILE // 6
            box = (max(0, y0 - m), min(h, y1 + m), max(0, x0 - m), min(w, x1 + m))
        return box, min(VIEW_W / (box[3] - box[2]), VIEW_H / (box[1] - box[0]))

    def render(self):
        (Y0, Y1, X0, X1), s = self.view()
        out = cv2.resize(self.img[Y0:Y1, X0:X1], None, fx=s, fy=s,
                         interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        if not self.overview:                  # dim everything outside the current tile
            y0, y1, x0, x1 = self.tiles[self.i]
            a, b = (int((x0 - X0) * s), int((y0 - Y0) * s)), (int((x1 - X0) * s), int((y1 - Y0) * s))
            dim = (out * 0.45).astype(np.uint8)
            dim[a[1]:b[1], a[0]:b[0]] = out[a[1]:b[1], a[0]:b[0]]
            out = dim
            cv2.rectangle(out, a, b, (0, 255, 255), 1)
        r = 3 if self.overview else max(3, int(2.5 * s))
        for x, y in self.points:
            if X0 <= x < X1 and Y0 <= y < Y1:
                c = (int((x - X0) * s), int((y - Y0) * s))
                cv2.circle(out, c, r, (0, 0, 0), -1)
                cv2.circle(out, c, r - 1, (0, 0, 255), -1)
        where = "overview" if self.overview else f"tile {self.i + 1}/{len(self.tiles)}"
        text = f"{len(self.points)} heads | {where} | n next  b back  o overview  u undo  q next frame"
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return out

    def on_mouse(self, event, x, y, flags, _=None):
        (Y0, Y1, X0, X1), s = self.view()
        px, py = X0 + x / s, Y0 + (y) / s
        remove = event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_SHIFTKEY)
        if remove and self.points:
            d = [math.hypot(px - a, py - b) for a, b in self.points]
            k = int(np.argmin(d))
            if d[k] * s < 20:                  # within 20 screen pixels
                self.history.append(("remove", self.points.pop(k)))
                self.save(self.points)
        elif event == cv2.EVENT_LBUTTONDOWN and y > 30:
            p = (round(px, 1), round(py, 1))
            self.points.append(p)
            self.history.append(("add", p))
            self.save(self.points)

    def undo(self):
        if self.history:
            kind, p = self.history.pop()
            self.points.remove(p) if kind == "add" else self.points.append(p)
            self.save(self.points)

    def key(self, k):
        """Returns "next" (next frame), "stop", or None (keep going)."""
        if k in (ord("n"), ord(" ")):
            self.overview, self.i = False, min(self.i + 1, len(self.tiles) - 1)
        elif k == ord("b"):
            self.overview, self.i = False, max(self.i - 1, 0)
        elif k == ord("o"):
            self.overview = not self.overview
        elif k == ord("u"):
            self.undo()
        elif k == ord("q"):
            return "next"
        elif k == 27:
            return "stop"
        return None


def prefill_points(img):
    sys.path.insert(0, "crowd/lessons")
    import torch
    from p2pnet_net import find_heads, load_p2pnet
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = load_p2pnet("crowd/models/p2pnet_shtech_a.pth", device)
    points, _ = find_heads(net, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return [(round(float(x), 1), round(float(y), 1)) for x, y in points]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--seconds", type=float, nargs="+")
    ap.add_argument("--every", type=float, help="take a frame every N seconds")
    ap.add_argument("--prefill", action="store_true", help="start from P2PNet's points (training frames only)")
    args = ap.parse_args()
    video = args.video.expanduser()
    if not video.exists():
        sys.exit(f"no such video: {video}")
    if args.seconds:
        seconds = args.seconds
    elif args.every:
        seconds = list(np.arange(0, video_seconds(video), args.every).round(1))
    else:
        sys.exit("give --seconds or --every")

    out_dir = LABELS / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    for sec in seconds:
        name = f"{video.stem}_{sec:g}s"
        img_path, json_path = out_dir / f"{name}.jpg", out_dir / f"{name}.json"
        if img_path.exists():
            img = cv2.imread(str(img_path))
        else:
            img = grab(video, sec)
            if img is None:
                print(f"{sec}s: cannot read frame, skipped")
                continue
            cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if json_path.exists():
            points = json.loads(json_path.read_text())["points"]
        else:
            points = prefill_points(img) if args.prefill else []

        def save(pts, sec=sec, img_path=img_path, json_path=json_path):
            json_path.write_text(json.dumps({"image": img_path.name, "video": str(video), "second": sec,
                                             "prefilled": args.prefill, "points": [list(p) for p in pts]}))

        save(points)
        lab = Labeler(img, points, save)
        cv2.setMouseCallback(WIN, lab.on_mouse)
        print(f"{name}: {len(lab.tiles)} tiles, starting with {len(points)} points")
        while True:
            cv2.imshow(WIN, lab.render())
            action = lab.key(cv2.waitKey(30) & 0xFF)
            if action:
                break
        print(f"{name}: saved {len(lab.points)} heads -> {json_path}")
        if action == "stop":
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
