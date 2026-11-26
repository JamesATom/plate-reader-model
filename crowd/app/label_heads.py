"""Crowd tool: click every head in video frames. The points become our true counts (to measure the
models) and later our training data (to fine-tune P2PNet).

What to do: click the middle of EVERY person's head you can see. That's all.

    left click    add a head
    right click   remove the nearest head   (or shift + click)
    Z             zoom in 3x where the mouse is / zoom back out (for small far-away heads)
    U             undo
    Q             this frame is done -> next frame        Esc   stop (everything is saved)

Run:  uv run python crowd/app/label_heads.py VIDEO --seconds 7 20 36    (chosen moments)
      uv run python crowd/app/label_heads.py VIDEO --every 5            (a frame every 5 seconds)
      add --prefill to start from P2PNet's dots and only correct them (faster; use it only for
      TRAINING videos: for test videos it would push the labels towards P2PNet's answer)

Saved after every click: crowd/data/labels/<video>/<video>_<second>s.jpg (full-size frame)
                         and .json {"image", "video", "second", "points": [[x, y], ...]}
Re-running the same command continues where you stopped.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

WIN = "label heads"
VIEW_W, VIEW_H = 1400, 900         # window size
ZOOM = 3
BAR = 64                           # help bar height (clicks there are ignored)
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
    def __init__(self, img, points, save, title="", region=None):
        self.img, self.points, self.save, self.title = img, [tuple(p) for p in points], save, title
        self.region = region                   # [x0, y0, x1, y1]: only heads inside were labelled (yellow box)
        self.history = []                      # ("add", point) or ("remove", point)
        self.zoom_at = None                    # frame point the zoomed view is centred on, or None
        self.mouse = (img.shape[1] / 2, img.shape[0] / 2)

    def view(self):
        """The part of the frame on screen (y0, y1, x0, x1) and its enlargement factor."""
        h, w = self.img.shape[:2]
        if self.zoom_at is None:
            box = (0, h, 0, w)
        else:
            rh, rw = h / ZOOM, w / ZOOM
            cx = min(max(self.zoom_at[0], rw / 2), w - rw / 2)
            cy = min(max(self.zoom_at[1], rh / 2), h - rh / 2)
            box = (int(cy - rh / 2), int(cy + rh / 2), int(cx - rw / 2), int(cx + rw / 2))
        return box, min(VIEW_W / (box[3] - box[2]), (VIEW_H - BAR) / (box[1] - box[0]))

    def to_frame(self, x, y):
        (Y0, _, X0, _), s = self.view()
        return X0 + x / s, Y0 + (y - BAR) / s

    def render(self):
        (Y0, Y1, X0, X1), s = self.view()
        pic = cv2.resize(self.img[Y0:Y1, X0:X1], None, fx=s, fy=s,
                         interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        if self.region:
            rx0, ry0, rx1, ry1 = self.region
            cv2.rectangle(pic, (int((rx0 - X0) * s), int((ry0 - Y0) * s)), (int((rx1 - X0) * s) - 1, int((ry1 - Y0) * s) - 1),
                          (0, 255, 255), 2)
        r = max(3, int(1.5 * s))
        for x, y in self.points:
            if X0 <= x < X1 and Y0 <= y < Y1:
                c = (int((x - X0) * s), int((y - Y0) * s))
                cv2.circle(pic, c, r + 1, (0, 0, 0), -1)
                cv2.circle(pic, c, r, (0, 0, 255), -1)
        bar = np.zeros((BAR, max(pic.shape[1], 900), 3), np.uint8)
        state = "ZOOMED IN (Z = zoom out)" if self.zoom_at else "Z = zoom in where the mouse is"
        cv2.putText(bar, f"{self.title}   {len(self.points)} heads   |   {state}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(bar, "Click EVERY head.  Right-click = remove   U = undo   Q = frame done, next   Esc = stop",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        pic = cv2.copyMakeBorder(pic, 0, 0, 0, bar.shape[1] - pic.shape[1], cv2.BORDER_CONSTANT)
        return np.vstack([bar, pic])

    def on_mouse(self, event, x, y, flags, _=None):
        if y < BAR:
            return
        px, py = self.to_frame(x, y)
        self.mouse = (px, py)
        _, s = self.view()
        remove = event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_SHIFTKEY)
        if remove and self.points:
            d = [math.hypot(px - a, py - b) for a, b in self.points]
            k = int(np.argmin(d))
            if d[k] * s < 20:                  # within 20 screen pixels
                self.history.append(("remove", self.points.pop(k)))
                self.save(self.points)
        elif event == cv2.EVENT_LBUTTONDOWN:
            h, w = self.img.shape[:2]
            if 0 <= px < w and 0 <= py < h:
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
        if k in (ord("z"), ord("Z")):
            self.zoom_at = None if self.zoom_at else self.mouse
        elif k in (ord("u"), ord("U")):
            self.undo()
        elif k in (ord("q"), ord("Q"), 13):
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
    ap.add_argument("--prefill", action="store_true", help="start from P2PNet's points (training videos only)")
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
    for n, sec in enumerate(seconds, 1):
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
        meta = json.loads(json_path.read_text()) if json_path.exists() else {"prefilled": args.prefill}
        points = meta.get("points") or (prefill_points(img) if args.prefill else [])

        def save(pts, sec=sec, img_path=img_path, json_path=json_path, meta=meta):
            # keep extra fields (e.g. "region": only that box was labelled, "labelled_by")
            meta.update({"image": img_path.name, "video": str(video), "second": sec, "points": [list(p) for p in pts]})
            json_path.write_text(json.dumps(meta))

        save(points)
        lab = Labeler(img, points, save, title=f"Frame {n}/{len(seconds)} ({sec:g}s)", region=meta.get("region"))
        cv2.setMouseCallback(WIN, lab.on_mouse)
        while True:
            cv2.imshow(WIN, lab.render())
            action = lab.key(cv2.waitKey(30) & 0xFF)
            if action:
                break
        print(f"frame {n}/{len(seconds)} ({sec:g}s): {len(lab.points)} heads saved")
        if action == "stop":
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
