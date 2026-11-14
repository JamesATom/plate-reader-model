"""Vehicle lesson 05: link the SAME car across the 4 RoundaboutHD cameras (the real re-ID job).

Lessons 01-04 asked "which gallery crop is this car?". A real system asks something else:
  a car LEFT camera 1 at 10:02:15. Which car that ENTERED camera 2 in the next minutes is it?
So we work with whole tracks (one car's passage through one camera), not single crops:

  1. tracks      every (camera, car) passage from the ground truth (perfect tracking, so this lesson
                 measures ONLY the cross-camera step; the app adds our own detector + tracker)
  2. appearance  8 crops per track, the bigger half of its boxes, spread in time -> re-ID model ->
                 mean vector = the track's "fingerprint"
  3. candidates  the cameras are a chain 1 - 2 - 3 - 4: a car leaving camera a can only appear next in
                 camera a-1 or a+1, and only 0..180 s later
  4. linking     best cosine first; each track gets at most one "next" and one "previous" track;
                 a link needs cosine >= threshold

Honest exam: the fine-tuned models saw the group-A cars (lesson 02), so links are scored on group-B cars
only; the threshold is picked on group A (best F1) and then used unchanged on group B.
Baseline "time only": no appearance, link the candidate whose travel time is closest to the typical one.

Run:  uv run python vehicle/lessons/05_link_cameras.py
"""

import importlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "vehicle/lessons")
from fastreid_net import load_vehicle_reid  # noqa: E402

lesson01 = importlib.import_module("01_reid_test")
lesson02 = importlib.import_module("02_finetune")

DATA = Path("vehicle/data/RoundaboutHD")
CACHE = Path("vehicle/output/v05_track_vectors.npz")
FPS, CROPS, WINDOW_S = 15, 8, 180
MODELS = {"VeRi only": None,
          "+ RoundaboutHD": "vehicle/models/roundabout_finetuned.pt",
          "+ RoundaboutHD + Uzbek": "vehicle/models/roundabout_uzbek_finetuned.pt"}
DEVICE = lesson01.DEVICE


def load_tracks():
    """(cam, gt_id) -> {frame: (x1, y1, x2, y2)}"""
    tracks = defaultdict(dict)
    for line in open(DATA / "Multi_CAM_Ground_Turth.txt"):
        c, i, f, x, y, w, h = line.split()[:7]
        x, y, w, h = map(float, (x, y, w, h))
        tracks[(int(c), int(i))][int(f)] = (x, y, x + w, y + h)
    return dict(tracks)


def reid_to_gt():
    """ReID-subset car id -> ground-truth id. The two files number cars differently, so each ReID crop
    is matched to the GT box of the same camera + frame with the closest width/height; majority vote."""
    by_cf = defaultdict(list)
    for (c, i), boxes in load_tracks().items():
        for f, (x1, y1, x2, y2) in boxes.items():
            by_cf[(c, f)].append((i, x2 - x1, y2 - y1))
    votes = defaultdict(Counter)
    for folder in ("train_images", "query_images", "test_images"):
        for p in sorted((lesson01.ROOT / folder).glob("*.jpg"))[::5]:
            car, cam, frame = lesson01.parse(p)
            cands = by_cf.get((int(cam[-2:]), frame))
            if cands:
                h, w = cv2.imread(str(p)).shape[:2]
                i, gw, gh = min(cands, key=lambda t: abs(t[1] - w) + abs(t[2] - h))
                if abs(gw - w) + abs(gh - h) < 6:
                    votes[car][i] += 1
    return {car: v.most_common(1)[0][0] for car, v in votes.items()}


def pick_frames(boxes):
    """CROPS frames from the bigger half of the boxes (closer = more detail), spread over time"""
    area = {f: (b[2] - b[0]) * (b[3] - b[1]) for f, b in boxes.items()}
    big = sorted(f for f in boxes if area[f] >= np.median(list(area.values())))
    return [big[int(k)] for k in np.linspace(0, len(big) - 1, min(CROPS, len(big)))]


def embed_tracks(tracks):
    nets = {}
    for name, w in MODELS.items():
        net = load_vehicle_reid("vehicle/models/veri_sbs_R50-ibn.pth")
        if w:
            net.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        nets[name] = net.to(DEVICE).eval()
    keys = sorted(tracks)
    vecs = {name: np.zeros((len(keys), 2048), np.float32) for name in MODELS}
    for cam in sorted({c for c, _ in keys}):
        wanted = defaultdict(list)                      # frame -> [(track index, box)]
        for k, key in enumerate(keys):
            if key[0] == cam:
                for f in pick_frames(tracks[key]):
                    wanted[f].append((k, tracks[key][f]))
        cap = cv2.VideoCapture(str(DATA / f"imagesc{cam:03d}" / "video.mp4"))
        batch, owners = [], []

        def flush():
            x = torch.from_numpy(np.stack(batch).astype(np.float32)).permute(0, 3, 1, 2).to(DEVICE)
            for name, net in nets.items():
                v = net(x).cpu().numpy()
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                for k, row in zip(owners, v):
                    vecs[name][k] += row
            batch.clear()
            owners.clear()

        last = max(wanted)
        for f in range(last + 1):
            if f not in wanted:
                cap.grab()
                continue
            ok, frame = cap.read()
            if not ok:
                break
            for k, (x1, y1, x2, y2) in wanted[f]:
                crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                if crop.size:
                    batch.append(cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (256, 256)))
                    owners.append(k)
            if len(batch) >= 64:
                flush()
        if batch:
            flush()
        print(f"  camera {cam}: {sum(len(v) for v in wanted.values())} crops embedded", flush=True)
    for name in MODELS:
        vecs[name] /= np.linalg.norm(vecs[name], axis=1, keepdims=True) + 1e-9
    return keys, vecs


def candidates(keys, tracks):
    """(a, b, gap seconds) for every pair where b could be the NEXT camera passage after a"""
    start = {k: min(tracks[k]) / FPS for k in keys}
    end = {k: max(tracks[k]) / FPS for k in keys}
    out = []
    for a, ka in enumerate(keys):
        for b, kb in enumerate(keys):
            if abs(ka[0] - kb[0]) == 1 and 0 <= start[kb] - end[ka] <= WINDOW_S:
                out.append((a, b, start[kb] - end[ka]))
    return out


def link(cands, score, threshold):
    """greedy, best score first; each track: at most one next and one previous"""
    has_next, has_prev, links = set(), set(), []
    for a, b, s in sorted(((a, b, score(a, b, g)) for a, b, g in cands), key=lambda t: -t[2]):
        if s < threshold:
            break
        if a not in has_next and b not in has_prev:
            links.append((a, b))
            has_next.add(a)
            has_prev.add(b)
    return links


def score_links(links, true_links, ids, group):
    """precision / recall / F1 counted on links that touch a car of `group`"""
    pred = [(a, b) for a, b in links if ids[a] in group or ids[b] in group]
    true = {(a, b) for a, b in true_links if ids[a] in group}
    good = sum((a, b) in true for a, b in pred)
    p, r = good / max(1, len(pred)), good / max(1, len(true))
    return p, r, 2 * p * r / max(1e-9, p + r), len(true)


def main():
    torch.set_grad_enabled(False)
    tracks = load_tracks()
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=False)
        keys = [tuple(k) for k in d["keys"]]
        vecs = {name: d[f"v{n}"] for n, name in enumerate(MODELS)}
    else:
        keys, vecs = embed_tracks(tracks)
        np.savez(CACHE, keys=np.array(keys), **{f"v{n}": vecs[name] for n, name in enumerate(MODELS)})
    ids = [k[1] for k in keys]

    mapping = reid_to_gt()
    _, group_b_reid = lesson02.split_cars()
    group_b = {mapping[c] for c in group_b_reid if c in mapping}
    group_a = set(ids) - group_b
    print(f"{len(keys)} tracks, {len(set(ids))} cars | ReID ids matched to GT: {len(mapping)} | "
          f"exam cars (group B, never trained on): {len(group_b)}")

    cands = candidates(keys, tracks)
    true_links = [(a, b) for a, b, _ in cands if ids[a] == ids[b]]
    print(f"candidate pairs (next camera, 0-{WINDOW_S} s): {len(cands)}, true ones: {len(true_links)} "
          f"(= {len(cands) / max(1, len(true_links)):.1f} candidates per true link)")

    # time only: typical travel time per camera pair, learned from group-A cars
    gaps = defaultdict(list)
    for a, b, g in cands:
        if ids[a] == ids[b] and ids[a] in group_a:
            gaps[(keys[a][0], keys[b][0])].append(g)
    typical = {k: np.median(v) for k, v in gaps.items()}
    gap_of = {(a, b): g for a, b, g in cands}

    rows = [("time only", lambda a, b, g: -abs(g - typical.get((keys[a][0], keys[b][0]), 40)) / 60)]
    rows += [(name, (lambda v: lambda a, b, g: float(v[a] @ v[b]))(vecs[name])) for name in MODELS]

    print(f"\n{'method':<24} {'threshold':>9}   {'precision':>9} {'recall':>7} {'F1':>6}   (group B, {score_links([], true_links, ids, group_b)[3]} true links)")
    for name, score in rows:
        values = sorted({round(score(a, b, gap_of[(a, b)]), 3) for a, b, _ in cands})
        best_t = max(values[::max(1, len(values) // 200)],
                     key=lambda t: score_links(link(cands, score, t), true_links, ids, group_a)[2])
        p, r, f1, _ = score_links(link(cands, score, best_t), true_links, ids, group_b)
        print(f"{name:<24} {best_t:>9.3f}   {p:>9.1%} {r:>7.1%} {f1:>6.1%}")


if __name__ == "__main__":
    main()
