"""Multi-camera vehicle tracking app: the same car gets the SAME number in every camera.

For each camera video (synchronised):
  1. detect   RF-DETR Medium (COCO) finds cars / buses / trucks, 5 times per second
              (saved once, so the next steps can be re-run in seconds)
  2. track    "ByteTrack" idea: predict where each car moves (constant speed), match confident boxes
              first (Hungarian, IoU), then let weak boxes (half-hidden car) continue existing tracks;
              a lost track waits LOST_S seconds at its predicted place. Pieces of one car are stitched.
              Ignored: parked cars (never move), cars outside the road area of the camera.
  3. describe 8 crops per track (the bigger half of its boxes) -> our re-ID model -> mean = fingerprint
Then across cameras (lesson 05):
  4. link     cameras are a chain 1-2-3-4; a track can continue in the next camera 0..180 s later.
              score = cosine - TIME_W * (how unusual the travel time is); best first,
              each track at most one next / one previous, score >= threshold
  5. number   linked tracks share one global number

Honest tuning: the road areas, typical travel times, TIME_W and the threshold all come from group-A cars
(seen in training); the score is on group-B cars only (never used anywhere).

Outputs (vehicle/output/):
  multicam_tracks.csv   global_id, camera, start_s, end_s, frames, next camera
  multicam.mp4          the 4 cameras in a 2x2 grid, boxes with global numbers (first --video-minutes)

Run:  uv run python vehicle/app/multicam.py                     (first run ~25 min on an M4: detection)
      uv run python vehicle/app/multicam.py --tune              (re-pick TIME_W and threshold on group A)
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
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "vehicle/lessons")
from fastreid_net import load_vehicle_reid  # noqa: E402

lesson05 = importlib.import_module("05_link_cameras")

DATA = Path("vehicle/data/RoundaboutHD")
OUT = Path("vehicle/output")
FPS, STEP = 15, 3                 # video is 15 fps, we look at every 3rd frame = 5 per second
VEHICLES = {3, 6, 8}              # COCO ids: car, bus, truck
LOW_CONF, HIGH_CONF, MIN_W = 0.15, 0.4, 40
IOU_HIGH, IOU_LOW, LOST_S = 0.2, 0.3, 5
EMBED_MIN_W = 60
MIN_LEN = 5                       # tracks seen fewer times are noise
STITCH_S, STITCH_COS = 8, 0.5     # join track pieces: at most 8 s apart and similar looking
ROI_CELL = 64                     # road-area grid cell, pixels
ZONE_CELL = 128                   # entry / exit zone grid cell, pixels
ZONE_GROW = int(os.environ.get("ZONE_GROW", 1))   # zones = cells where group-A trips left / entered, grown by this many cells
SETTINGS = {"roundabout_uzbek_finetuned.pt": (0.29, 0.6)}   # (threshold, TIME_W), picked with --tune (2-fold cross-validation)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def detect_camera(cam, detector, last_frame):
    """-> {frame: [(x1, y1, x2, y2, conf), ...]} vehicles only, also weak ones (>= LOW_CONF)"""
    cache = OUT / f"multicam_det_cam{cam}.json"
    if cache.exists():
        d = json.load(open(cache))
        if d["last_frame"] >= last_frame:
            return {int(f): v for f, v in d["frames"].items() if int(f) < last_frame}
    cap = cv2.VideoCapture(str(DATA / f"imagesc{cam:03d}" / "video.mp4"))
    out, t0 = {}, time.perf_counter()
    for f in range(0, last_frame, STEP):
        for _ in range(STEP - 1 if f else 0):
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            break
        det = detector.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), threshold=LOW_CONF)
        out[f] = [[*map(float, b), float(c)] for b, k, c in zip(det.xyxy, det.class_id, det.confidence)
                  if int(k) in VEHICLES and b[2] - b[0] >= MIN_W]
        if f % (FPS * 60) == 0:
            print(f"  camera {cam}: detecting {f / FPS / 60:.0f}/{last_frame / FPS / 60:.0f} min "
                  f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    json.dump({"last_frame": last_frame, "frames": out}, open(cache, "w"))
    return out


def dedup(boxes):
    """class-agnostic NMS: RF-DETR (no NMS inside) gives one car several boxes, e.g. "car" + "truck" for a
    pickup or a weak copy at low confidence. Keep the surest; drop boxes that overlap it (IoU > 0.6) or
    lie mostly inside it (> 85% of the smaller box)."""
    kept = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        dup = False
        for k in kept:
            inter = max(0, min(b[2], k[2]) - max(b[0], k[0])) * max(0, min(b[3], k[3]) - max(b[1], k[1]))
            sb, sk = (b[2] - b[0]) * (b[3] - b[1]), (k[2] - k[0]) * (k[3] - k[1])
            if inter / (sb + sk - inter) > 0.6 or inter / min(sb, sk) > 0.85:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def track(dets):
    """ByteTrack-style tracking -> [{frames: {frame: box}}]"""
    active, done, max_lost = [], [], LOST_S * FPS // STEP

    def predicted(t):
        return t["last"] + t["vel"] * (t["lost"] + 1)

    def match(tracks, boxes, min_iou):
        if not tracks or not boxes:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        cost = np.array([[1 - iou(predicted(t), b) for b in boxes] for t in tracks])
        rows, cols = linear_sum_assignment(cost)
        pairs = [(r, c) for r, c in zip(rows, cols) if 1 - cost[r, c] >= min_iou]
        mt, mb = {r for r, _ in pairs}, {c for _, c in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(boxes)) if j not in mb]

    def update(t, box, f):
        box = np.array(box[:4])
        t["vel"] = 0.5 * t["vel"] + 0.5 * (box - t["last"]) / (t["lost"] + 1)   # smoothed speed per look
        t["last"], t["lost"] = box, 0
        t["frames"][f] = box.tolist()

    for f in sorted(dets):
        boxes = dedup(dets[f])
        high = [d for d in boxes if d[4] >= HIGH_CONF]
        low = [d for d in boxes if d[4] < HIGH_CONF]
        pairs, rest, new = match(active, high, IOU_HIGH)            # 1. confident boxes, all tracks
        for r, c in pairs:
            update(active[r], high[c], f)
        recent = [active[i] for i in rest if active[i]["lost"] == 0]
        pairs2, _, _ = match(recent, low, IOU_LOW)                  # 2. weak boxes continue recent tracks
        for r, c in pairs2:
            update(recent[r], low[c], f)
        for t in active:
            if f not in t["frames"]:
                t["lost"] += 1
        for c in new:                                               # 3. new tracks from confident boxes only
            active.append({"frames": {f: high[c][:4]}, "last": np.array(high[c][:4]), "vel": np.zeros(4), "lost": 0})
        done += [t for t in active if t["lost"] > max_lost]
        active = [t for t in active if t["lost"] <= max_lost]
    return [{"frames": t["frames"]} for t in done + active]


def road_masks(group_b):
    """per camera: grid cells where group-A cars drive (stands in for an area an operator would draw once)"""
    masks = {}
    for (c, i), boxes in lesson05.load_tracks().items():
        if i in group_b:
            continue
        m = masks.setdefault(c, np.zeros((2160 // ROI_CELL + 1, 3840 // ROI_CELL + 1), np.uint8))
        for x1, y1, x2, y2 in boxes.values():
            m[min(int(y2) // ROI_CELL, m.shape[0] - 1), min(int((x1 + x2) / 2) // ROI_CELL, m.shape[1] - 1)] = 1
    return {c: cv2.dilate(m, np.ones((5, 5), np.uint8)) for c, m in masks.items()}


def keep(t, mask):
    """long enough, moving, and mostly on the road area"""
    b = list(t["frames"].values())
    if len(b) < MIN_LEN:
        return False
    centres = np.array([((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in b])
    if np.linalg.norm(centres.max(0) - centres.min(0)) < 0.5 * np.median([x2 - x1 for x1, _, x2, _ in b]):
        return False                                            # parked: never moved half its width
    on_road = [mask[min(int(y2) // ROI_CELL, mask.shape[0] - 1), min(int((x1 + x2) / 2) // ROI_CELL, mask.shape[1] - 1)]
               for x1, _, x2, y2 in b]
    return np.mean(on_road) >= 0.5


def embed(cam, tracks, net):
    """8 crops per track (bigger half of its boxes, >= EMBED_MIN_W wide) -> mean re-ID vector"""
    wanted = defaultdict(list)
    for k, t in enumerate(tracks):
        wide = {f: b for f, b in t["frames"].items() if b[2] - b[0] >= EMBED_MIN_W}
        for f in lesson05.pick_frames(wide) if wide else []:
            wanted[f].append((k, wide[f]))
    sums = np.zeros((len(tracks), 2048), np.float32)
    cap = cv2.VideoCapture(str(DATA / f"imagesc{cam:03d}" / "video.mp4"))
    batch, owners = [], []

    def flush():
        x = torch.from_numpy(np.stack(batch).astype(np.float32)).permute(0, 3, 1, 2).to(DEVICE)
        v = net(x).cpu().numpy()
        for k, row in zip(owners, v / np.linalg.norm(v, axis=1, keepdims=True)):
            sums[k] += row
        batch.clear()
        owners.clear()

    for f in range(max(wanted) + 1 if wanted else 0):
        if f not in wanted:
            cap.grab()
            continue
        ok, frame = cap.read()
        if not ok:
            break
        for k, (x1, y1, x2, y2) in wanted[f]:
            batch.append(cv2.resize(cv2.cvtColor(frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)],
                                                 cv2.COLOR_BGR2RGB), (256, 256)))
            owners.append(k)
        if len(batch) >= 64:
            flush()
    if batch:
        flush()
    kept = []
    for k, t in enumerate(tracks):
        if sums[k].any():
            kept.append({"frames": t["frames"], "vec": sums[k] / np.linalg.norm(sums[k])})
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


def exam_cars():
    """group B as ground-truth ids: the cars never used in any training (lesson 02 split)"""
    mapping = lesson05.reid_to_gt()
    _, group_b_reid = lesson05.lesson02.split_cars()
    return {mapping[c] for c in group_b_reid if c in mapping}


def travel_times(group_b):
    """typical seconds from leaving camera a to entering camera b, from group-A cars: (median, spread)"""
    gaps = defaultdict(list)
    per_car = defaultdict(list)
    for (c, i), b in lesson05.load_tracks().items():
        if i not in group_b:
            per_car[i].append((min(b), max(b), c))
    for seq in per_car.values():
        seq.sort()
        for (s1, e1, c1), (s2, e2, c2) in zip(seq, seq[1:]):
            if abs(c1 - c2) == 1:
                gaps[(c1, c2)].append((s2 - e1) / FPS)
    return {k: (np.median(v), np.percentile(v, 75) - np.percentile(v, 25)) for k, v in gaps.items()}


def foot_cell(box):
    """grid cell under the middle of the box's bottom edge (where the car touches the road)"""
    return int(box[3]) // ZONE_CELL, int((box[0] + box[2]) / 2) // ZONE_CELL


def learn_zones(group_b):
    """per camera pair (a, b): where cars LEAVE camera a and where they ENTER camera b on that trip,
    learned from group-A cars (an operator would draw these once per camera)"""
    exits, entries = defaultdict(set), defaultdict(set)
    per_car = defaultdict(list)
    for (c, i), b in lesson05.load_tracks().items():
        if i not in group_b:
            per_car[i].append((min(b), max(b), c, b))
    for seq in per_car.values():
        seq.sort(key=lambda t: t[0])
        for (s1, e1, c1, b1), (s2, e2, c2, b2) in zip(seq, seq[1:]):
            if abs(c1 - c2) == 1:
                exits[(c1, c2)].add(foot_cell(b1[e1]))
                entries[(c1, c2)].add(foot_cell(b2[s2]))

    def grow(cells):
        r = range(-ZONE_GROW, ZONE_GROW + 1)
        return {(y + dy, x + dx) for y, x in cells for dy in r for dx in r}
    return {p: grow(c) for p, c in exits.items()}, {p: grow(c) for p, c in entries.items()}


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
    return gt, owner


def loose_owners(keys, tracks, gt):
    """like gt_owners but forgiving (IoU >= 0.3 on any frames): for tracks that only partly match a
    ground-truth box (cut at the picture edge, longer than the labelled part of the road)"""
    by_cf = defaultdict(list)
    for (c, i), boxes in gt.items():
        for f, b in boxes.items():
            by_cf[(c, f)].append((i, b))
    out = []
    for (cam, _), t in zip(keys, tracks):
        votes = Counter()
        for f, b in t["frames"].items():
            hit = max(((iou(b, g), i) for i, g in by_cf.get((cam, f), [])), default=(0, None))
            if hit[0] >= 0.3:
                votes[hit[1]] += 1
        out.append(votes.most_common(1)[0][0] if votes else None)
    return out


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
        loose = loose_owners(keys, tracks, gt)
        same = sum(owner[a] != owner[b] and (owner[a] is None or owner[b] is None)
                   and (owner[a] or loose[a]) is not None and (owner[a] or loose[a]) == (owner[b] or loose[b])
                   for a, b in pred)
        print(f"    of those {to_none}: {same} join the SAME car after all (its track only partly matches the "
              f"ground-truth box) -> precision counting them right: {(good + same) / max(1, len(pred)):.1%}")
        print(f"  real camera-to-camera trips: {len(transitions)}, found: {len(transitions & made)} -> recall {r:.1%}"
              f"   F1 {f1:.1%}")
    return p, r, f1, (good, len(pred), len(transitions & made), len(transitions))


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
    ap.add_argument("--time-w", type=float)
    ap.add_argument("--video-minutes", type=float, default=3)
    ap.add_argument("--tune", action="store_true", help="pick threshold and TIME_W by cross-validation")
    ap.add_argument("--no-zones", action="store_true", help="do not use entry / exit zones")
    args = ap.parse_args()
    threshold, time_w = SETTINGS.get(Path(args.model).name, (0.6, 0.0))
    threshold = args.threshold if args.threshold is not None else threshold
    time_w = args.time_w if args.time_w is not None else time_w
    torch.set_grad_enabled(False)
    last_frame = int(args.minutes * 60 * FPS)
    group_b = exam_cars()
    masks = road_masks(group_b)

    detector = None
    net = load_vehicle_reid("vehicle/models/veri_sbs_R50-ibn.pth")
    net.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    net = net.to(DEVICE).eval()
    keys, tracks = [], []
    # cache of the described tracks: tracker / embedding settings + model in the name, so linking can be re-run in seconds
    tag = f"{args.minutes:g}min_{Path(args.model).stem}_c{LOW_CONF}-{HIGH_CONF}_iou{IOU_HIGH}-{IOU_LOW}_lost{LOST_S}_st{STITCH_S}-{STITCH_COS}_dedup"
    track_cache = OUT / f"multicam_tracks_{tag}.json"
    if track_cache.exists():
        saved = json.load(open(track_cache))
        keys = [tuple(k) for k in saved["keys"]]
        tracks = [{"frames": {int(f): b for f, b in t["frames"].items()}, "vec": np.array(t["vec"], np.float32)}
                  for t in saved["tracks"]]
    for cam in (1, 2, 3, 4) if not tracks else ():
        if detector is None and not (OUT / f"multicam_det_cam{cam}.json").exists():
            from rfdetr import RFDETRMedium
            detector = RFDETRMedium()
        raw = track(detect_camera(cam, detector, last_frame))
        kept = [t for t in raw if keep(t, masks[cam])]
        described = embed(cam, kept, net)
        joined = stitch(described)
        keys += [(cam, n) for n in range(len(joined))]
        tracks += joined
        print(f"camera {cam}: {len(raw)} tracks -> {len(kept)} moving on the road -> {len(joined)} after stitching",
              flush=True)
    if not track_cache.exists():
        json.dump({"keys": keys, "tracks": [{"frames": t["frames"], "vec": t["vec"].tolist()} for t in tracks]},
                  open(track_cache, "w"))

    vecs = np.stack([t["vec"] for t in tracks])
    cands = lesson05.candidates(keys, {k: t["frames"] for k, t in zip(keys, tracks)})
    all_cands = cands
    if not args.no_zones:     # a trip a -> b must leave a through its exit zone and enter b through its entry zone
        exits, entries = learn_zones(group_b)
        ends = [foot_cell(t["frames"][max(t["frames"])]) for t in tracks]
        starts = [foot_cell(t["frames"][min(t["frames"])]) for t in tracks]
        before = len(cands)
        cands = [(a, b, g) for a, b, g in cands
                 if ends[a] in exits.get((keys[a][0], keys[b][0]), ()) and starts[b] in entries.get((keys[a][0], keys[b][0]), ())]
        print(f"entry / exit zones: {before} -> {len(cands)} candidate pairs")
    typical = travel_times(group_b)

    def unusual(a, b, g):
        """minutes the travel time lies outside the usual range for this camera pair"""
        med, spread = typical.get((keys[a][0], keys[b][0]), (40, 30))
        return max(0.0, abs(g - med) - spread) / 60

    def scorer(w):
        return lambda a, b, g: float(vecs[a] @ vecs[b]) - w * unusual(a, b, g)

    gt, owner = gt_owners(keys, tracks)
    if args.tune:
        # The fine-tuned model has SEEN group-A cars: their similarities are inflated, so a threshold picked
        # there is too strict for new cars. Honest alternative: 2-fold cross-validation inside group B:
        # pick (threshold, TIME_W) on one half of the cars, score on the other half, then swap.
        cars = sorted(group_b)
        halves = [set(cars[0::2]), set(cars[1::2])]
        total, picked = np.zeros(4), []
        for pick, test in ((halves[0], halves[1]), (halves[1], halves[0])):
            best = max(((evaluate(keys, tracks, lesson05.link(cands, scorer(w), t), gt, owner, pick, quiet=True)[2], t, w)
                        for w in (0.0, 0.2, 0.4, 0.8) for t in np.arange(0.0, 0.81, 0.02)), key=lambda x: x[0])
            picked.append(best[1:])
            total += evaluate(keys, tracks, lesson05.link(cands, scorer(best[2]), best[1]), gt, owner, test, quiet=True)[3]
        p, r = total[0] / total[1], total[2] / total[3]
        print(f"cross-validated on group B: picked {[(round(float(t), 2), w) for t, w in picked]} -> "
              f"precision {p:.1%}  recall {r:.1%}  F1 {2 * p * r / (p + r):.1%}   <- the honest score")
        threshold = round(float(np.mean([t for t, _ in picked])), 2)
        time_w = float(np.mean([w for _, w in picked]))
        print(f"settings for use: threshold {threshold}, TIME_W {time_w} (mean of the two picks)")
    links = lesson05.link(cands, scorer(time_w), threshold)
    if os.environ.get("DUMP"):          # for error analysis
        np.savez(os.environ["DUMP"], keys=np.array(keys), vecs=vecs, owner=np.array([-1 if o is None else o for o in owner]),
                 all_cands=np.array(all_cands), cands=np.array(cands), links=np.array(links),
                 ends=np.array([max(t["frames"]) for t in tracks]), starts=np.array([min(t["frames"]) for t in tracks]),
                 group_b=np.array(sorted(group_b)), unusual=np.array([unusual(a, b, g) for a, b, g in cands]))
        json.dump([{str(f): b for f, b in t["frames"].items()} for t in tracks], open(os.environ["DUMP"] + ".json", "w"))

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
          f"(threshold {threshold}, TIME_W {time_w})")

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
