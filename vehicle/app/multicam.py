"""Multi-camera vehicle tracking app: the same car gets the SAME number in every camera.

For each camera video (synchronised):
  1. detect   RF-DETR Medium (COCO) finds cars / buses / trucks, 5 times per second
  2. track    a box is joined to the track whose last box overlaps it most (IoU), so one car's passage
              through one camera = one track. Tracks that do not move (parked cars) are ignored.
  3. describe every 3rd box of a track (>= 60 px wide) -> our re-ID model -> 2048 numbers;
              the track's fingerprint = mean of its bigger half
Then across cameras (lesson 05):
  4. link     cameras are a chain 1-2-3-4; a track can continue in the next camera 0..180 s later;
              best cosine first, each track at most one next / one previous, cosine >= THRESHOLD
  5. number   linked tracks share one global number

Outputs (vehicle/output/):
  multicam_tracks.csv   global_id, camera, start_s, end_s, frames, linked_to
  multicam.mp4          the 4 cameras in a 2x2 grid, boxes with global numbers (first --video-minutes)
  + a score against the RoundaboutHD ground truth (cars never used in training, group B)

Run:  uv run python vehicle/app/multicam.py                     (all 10 minutes, ~25 min on an M4)
      uv run python vehicle/app/multicam.py --minutes 2         (quick test)
      uv run python vehicle/app/multicam.py --model vehicle/models/roundabout_finetuned.pt
"""

import argparse
import bisect
import csv
import importlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "vehicle/lessons")
from fastreid_net import load_vehicle_reid  # noqa: E402

lesson05 = importlib.import_module("05_link_cameras")

DATA = Path("vehicle/data/RoundaboutHD")
OUT = Path("vehicle/output")
FPS, STEP = 15, 3                 # video is 15 fps, we look at every 3rd frame = 5 per second
VEHICLES = {3, 6, 8}              # COCO ids: car, bus, truck
DET_CONF, MIN_W = 0.4, 40
MATCH_IOU, MAX_MISSING = 0.1, 10  # tracking: min overlap; a track may miss 10 looks (2 s) before it ends
EMBED_EVERY, EMBED_MIN_W = 3, 60
MIN_LEN = 5                       # tracks seen fewer times are noise
STITCH_S, STITCH_COS = 8, 0.5     # join track pieces: at most 8 s apart and similar looking
THRESHOLD = {"roundabout_uzbek_finetuned.pt": 0.58}  # picked on group-A cars with --tune (never on the exam cars)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def track_camera(cam, detector, net, last_frame):
    """-> list of tracks {frames: {frame: box}, vecs: [(area, vector)]}"""
    cap = cv2.VideoCapture(str(DATA / f"imagesc{cam:03d}" / "video.mp4"))
    active, done = [], []
    t0 = time.perf_counter()
    for f in range(0, last_frame, STEP):
        for _ in range(STEP - 1 if f else 0):
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            break
        det = detector.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), threshold=DET_CONF)
        boxes = [tuple(map(float, b)) for b, c in zip(det.xyxy, det.class_id)
                 if int(c) in VEHICLES and b[2] - b[0] >= MIN_W]

        # match boxes to active tracks: best overlap first
        pairs = sorted(((iou(t["last"], b), ti, bi) for ti, t in enumerate(active) for bi, b in enumerate(boxes)),
                       reverse=True)
        used_t, used_b = set(), set()
        for v, ti, bi in pairs:
            if v < MATCH_IOU:
                break
            if ti not in used_t and bi not in used_b:
                used_t.add(ti)
                used_b.add(bi)
                active[ti]["frames"][f] = boxes[bi]
                active[ti]["last"], active[ti]["missing"] = boxes[bi], 0
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                active.append({"frames": {f: b}, "last": b, "missing": 0, "vecs": []})
        for ti, t in enumerate(active):
            if ti not in used_t and f not in t["frames"]:
                t["missing"] += 1

        # describe: every EMBED_EVERY-th box of each track
        crops, owners = [], []
        for t in active:
            if t["missing"] == 0 and (len(t["frames"]) - 1) % EMBED_EVERY == 0:
                x1, y1, x2, y2 = (int(v) for v in t["last"])
                if x2 - x1 >= EMBED_MIN_W:
                    crops.append(cv2.resize(cv2.cvtColor(frame[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2RGB), (256, 256)))
                    owners.append(t)
        if crops:
            x = torch.from_numpy(np.stack(crops).astype(np.float32)).permute(0, 3, 1, 2).to(DEVICE)
            v = net(x).cpu().numpy()
            for t, row, c in zip(owners, v / np.linalg.norm(v, axis=1, keepdims=True), crops):
                t["vecs"].append(((t["last"][2] - t["last"][0]) * (t["last"][3] - t["last"][1]), row.astype(np.float16)))

        done += [t for t in active if t["missing"] > MAX_MISSING]
        active = [t for t in active if t["missing"] <= MAX_MISSING]
        if f % (FPS * 60) == 0:
            print(f"  camera {cam}: {f / FPS / 60:.0f}/{last_frame / FPS / 60:.0f} min, "
                  f"{len(done) + len(active)} tracks ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    return done + active


def finish(tracks):
    """keep moving, long-enough tracks that have a fingerprint; fingerprint = mean of the bigger half"""
    kept = []
    for t in tracks:
        if len(t["frames"]) < MIN_LEN or not t["vecs"]:
            continue
        b = list(t["frames"].values())
        centres = np.array([((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in b])
        width = np.median([x2 - x1 for x1, _, x2, _ in b])
        if np.linalg.norm(centres.max(0) - centres.min(0)) < 0.5 * width:
            continue                                            # parked: never moved half its width
        areas = np.array([a for a, _ in t["vecs"]])
        v = np.mean([vec.astype(np.float32) for a, vec in t["vecs"] if a >= np.median(areas)], axis=0)
        kept.append({"frames": t["frames"], "vec": v / np.linalg.norm(v)})
    return kept


def stitch(tracks):
    """join pieces of one car's track inside one camera (the car was lost behind a sign / the island):
    piece B starts <= STITCH_S after piece A ends, near where A was heading, and looks like A.
    Best-looking pairs first; each piece gets at most one next and one previous piece."""
    def centre(box):
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    tracks = sorted(tracks, key=lambda t: min(t["frames"]))
    starts = [min(t["frames"]) for t in tracks]
    pairs = []
    for i, a in enumerate(tracks):
        fa = sorted(a["frames"])
        end, last = fa[-1], a["frames"][fa[-1]]
        f0 = fa[max(0, len(fa) - 4)]
        speed = (centre(last) - centre(a["frames"][f0])) / max(1, end - f0)      # pixels per frame
        for j in range(bisect.bisect_right(starts, end), bisect.bisect_right(starts, end + STITCH_S * FPS)):
            gap = starts[j] - end
            if np.linalg.norm(centre(tracks[j]["frames"][starts[j]]) - (centre(last) + speed * gap)) <= 1.5 * (last[2] - last[0]):
                sim = float(a["vec"] @ tracks[j]["vec"])
                if sim >= STITCH_COS:
                    pairs.append((sim, i, j))
    nxt, has_prev = {}, set()
    for _, i, j in sorted(pairs, reverse=True):
        if i not in nxt and j not in has_prev:
            nxt[i] = j
            has_prev.add(j)
    out = []
    for i in range(len(tracks)):
        if i in has_prev:
            continue
        frames, v = {}, np.zeros_like(tracks[i]["vec"])
        while i is not None:
            frames.update(tracks[i]["frames"])
            v += tracks[i]["vec"] * len(tracks[i]["frames"])
            i = nxt.get(i)
        out.append({"frames": frames, "vec": v / np.linalg.norm(v)})
    return out


def gt_owners(keys, tracks):
    """our track -> the ground-truth car it follows (IoU >= 0.5 on at least half of its frames), or None"""
    gt = lesson05.load_tracks()
    by_cf = defaultdict(list)
    for (c, i), boxes in gt.items():
        for f, b in boxes.items():
            by_cf[(c, f)].append((i, b))
    owner = []
    for (cam, _), t in zip(keys, tracks):
        votes = Counter()
        for f, b in t["frames"].items():
            hit = max(((iou(b, g), i) for i, g in by_cf.get((cam, f), [])), default=(0, None))
            if hit[0] >= 0.5:
                votes[hit[1]] += 1
        best = votes.most_common(1)
        owner.append(best[0][0] if best and best[0][1] >= 0.5 * len(t["frames"]) else None)
    mapping = lesson05.reid_to_gt()
    _, group_b_reid = lesson05.lesson02.split_cars()
    group_b = {mapping[c] for c in group_b_reid if c in mapping}
    return gt, owner, group_b


def evaluate(keys, tracks, links, gt, owner, group, quiet=False):
    """precision: share of links (touching a car of `group`) that join the same car;
    recall: share of real camera-to-camera trips of `group` cars that we linked"""
    last_frame = max(max(t["frames"]) for t in tracks)
    found = {(keys[n][0], owner[n]) for n in range(len(keys)) if owner[n] is not None}
    transitions = set()
    for x in group:
        seq = sorted((min(b), max(b), c) for (c, i), b in gt.items() if i == x and min(b) <= last_frame)
        for (s1, e1, c1), (s2, e2, c2) in zip(seq, seq[1:]):
            if abs(c1 - c2) == 1 and 0 <= (s2 - e1) / FPS <= lesson05.WINDOW_S and (c1, x) in found and (c2, x) in found:
                transitions.add((x, c1, c2))
    made = {(owner[a], keys[a][0], keys[b][0]) for a, b in links if owner[a] is not None and owner[a] == owner[b]}
    pred = [(a, b) for a, b in links if owner[a] in group or owner[b] in group]
    good = sum(owner[a] == owner[b] for a, b in pred)
    p, r = good / max(1, len(pred)), len(transitions & made) / max(1, len(transitions))
    f1 = 2 * p * r / max(1e-9, p + r)
    if not quiet:
        gt_passages = {k for k, b in gt.items() if min(b) <= last_frame and k[1] in group}
        to_none = sum(owner[a] is None or owner[b] is None for a, b in pred)
        print(f"\nscore vs ground truth (group B = {len(group)} cars never used in training):")
        print(f"  car passages found by detector + tracker: {len(found & gt_passages)}/{len(gt_passages)} "
              f"({len(found & gt_passages) / max(1, len(gt_passages)):.1%}), "
              f"{sum(o in group for o in owner) / max(1, len(found & gt_passages)):.2f} track pieces per passage")
        print(f"  our tracks that are not a ground-truth car (cars outside the labelled area, bits of tracks): "
              f"{sum(o is None for o in owner)}/{len(owner)}")
        print(f"  links made: {len(pred)}, right: {good} -> precision {p:.1%}   (wrong: {to_none} to a vehicle "
              f"the ground truth does not have, {len(pred) - good - to_none} to another car)")
        print(f"  real camera-to-camera trips: {len(transitions)}, found: {len(transitions & made)} -> recall {r:.1%}"
              f"   F1 {f1:.1%}")
    return p, r, f1


def render(keys, tracks, gid, minutes):
    colours = {}
    rng = np.random.default_rng(0)
    caps = {c: cv2.VideoCapture(str(DATA / f"imagesc{c:03d}" / "video.mp4")) for c in (1, 2, 3, 4)}
    at = defaultdict(list)                                    # (cam, frame) -> [(box, global id)]
    for k, t, g in zip(keys, tracks, gid):
        for f, b in t["frames"].items():
            at[(k[0], f)].append((b, g))
    out = cv2.VideoWriter(str(OUT / "multicam.mp4"), cv2.VideoWriter_fourcc(*"avc1"), FPS / STEP, (1920, 1080))
    for f in range(0, int(minutes * 60 * FPS), STEP):
        tiles = []
        for c in (1, 2, 3, 4):
            for _ in range(STEP - 1 if f else 0):
                caps[c].grab()
            ok, frame = caps[c].read()
            if not ok:
                frame = np.zeros((2160, 3840, 3), np.uint8)
            for (x1, y1, x2, y2), g in at.get((c, f), []):
                col = colours.setdefault(g, tuple(int(v) for v in rng.integers(60, 255, 3)))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 6)
                cv2.putText(frame, f"#{g}", (int(x1), int(y1) - 12), cv2.FONT_HERSHEY_SIMPLEX, 2.2, col, 6)
            tile = cv2.resize(frame, (960, 540))
            cv2.putText(tile, f"camera {c}   {f / FPS // 60:02.0f}:{f / FPS % 60:04.1f}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            tiles.append(tile)
        out.write(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    out.release()
    print(f"video: {OUT / 'multicam.mp4'} ({minutes} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--model", default="vehicle/models/roundabout_uzbek_finetuned.pt")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--video-minutes", type=float, default=3)
    ap.add_argument("--tune", action="store_true", help="pick the threshold on group-A cars")
    args = ap.parse_args()
    threshold = args.threshold or THRESHOLD.get(Path(args.model).name, 0.6)
    torch.set_grad_enabled(False)

    cache = OUT / f"multicam_tracks_{args.minutes:g}min_{Path(args.model).stem}.json"
    if cache.exists():
        saved = json.load(open(cache))
        keys = [tuple(k) for k in saved["keys"]]
        tracks = [{"frames": {int(f): b for f, b in t["frames"].items()}, "vec": np.array(t["vec"])} for t in saved["tracks"]]
    else:
        from rfdetr import RFDETRMedium
        detector = RFDETRMedium()
        net = load_vehicle_reid("vehicle/models/veri_sbs_R50-ibn.pth")
        net.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
        net = net.to(DEVICE).eval()
        keys, tracks = [], []
        for cam in (1, 2, 3, 4):
            kept = finish(track_camera(cam, detector, net, int(args.minutes * 60 * FPS)))
            keys += [(cam, n) for n in range(len(kept))]
            tracks += kept
            print(f"camera {cam}: {len(kept)} moving vehicle tracks", flush=True)
        json.dump({"keys": keys, "tracks": [{"frames": t["frames"], "vec": t["vec"].tolist()} for t in tracks]},
                  open(cache, "w"))

    before = len(tracks)
    stitched_keys, stitched = [], []
    for cam in (1, 2, 3, 4):
        joined = stitch([t for k, t in zip(keys, tracks) if k[0] == cam])
        stitched_keys += [(cam, n) for n in range(len(joined))]
        stitched += joined
    keys, tracks = stitched_keys, stitched
    print(f"stitching broken tracks: {before} -> {len(tracks)} tracks")
    vecs = np.stack([t["vec"] for t in tracks])
    cands = lesson05.candidates(keys, {k: t["frames"] for k, t in zip(keys, tracks)})
    score = lambda a, b, g: float(vecs[a] @ vecs[b])      # noqa: E731
    gt, owner, group_b = gt_owners(keys, tracks)
    if args.tune:     # pick the threshold on group-A cars (seen in training), never on the exam cars
        group_a = {o for o in owner if o is not None} - group_b
        grid = np.arange(0.40, 0.81, 0.02)
        f1s = [evaluate(keys, tracks, lesson05.link(cands, score, t), gt, owner, group_a, quiet=True)[2] for t in grid]
        threshold = float(grid[int(np.argmax(f1s))])
        print(f"threshold picked on group A: {threshold:.2f} (F1 there {max(f1s):.1%})")
    links = lesson05.link(cands, score, threshold)

    # global numbers: follow the links
    nxt = dict(links)
    has_prev = {b for _, b in links}
    gid, n = [0] * len(keys), 0
    for a in range(len(keys)):
        if a not in has_prev:
            n += 1
            while a is not None:
                gid[a] = n
                a = nxt.get(a)
    print(f"\n{len(keys)} tracks in 4 cameras -> {len(links)} camera-to-camera links -> {n} different vehicles "
          f"(threshold {threshold})")

    with open(OUT / "multicam_tracks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["global_id", "camera", "start_s", "end_s", "frames_seen", "next_camera_track"])
        for k in sorted(range(len(keys)), key=lambda k: (gid[k], min(tracks[k]["frames"]))):
            fr = tracks[k]["frames"]
            w.writerow([gid[k], keys[k][0], f"{min(fr) / FPS:.1f}", f"{max(fr) / FPS:.1f}", len(fr),
                        f"cam{keys[nxt[k]][0]}" if k in nxt else ""])
    print(f"table: {OUT / 'multicam_tracks.csv'}")

    evaluate(keys, tracks, links, gt, owner, group_b)
    if args.video_minutes:
        render(keys, tracks, gid, min(args.video_minutes, args.minutes))


if __name__ == "__main__":
    main()
