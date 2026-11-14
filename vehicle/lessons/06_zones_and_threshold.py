"""Vehicle lesson 06: three cheap ways to link cars across cameras better (no new training).

Same perfect (ground-truth) tracks and the same RoundaboutHD + Uzbek model as lesson 05.
  1. camera bias   subtract each camera's mean vector (colour / light of that camera).
                   Helps the base VeRi model (+2.7 F1) but not the fine-tuned ones: they already know
                   these cameras. Not used.
  2. travel time   score = cosine - 0.4 x minutes the trip lies outside the usual range of that pair
  3. zones         a car going a -> b leaves a through one area of the picture and enters b through
                   another. Learned from group-A trips (128 px grid, grown by 1 cell). 70k candidate
                   pairs -> 19k, 97% of the true trips kept.
  +  threshold     the model has SEEN group-A cars: same-car cosine there is 0.84 (median), on new cars
                   0.69. A threshold picked on group A is too strict for new cars. Honest fix: 2-fold
                   cross-validation inside group B (pick on half of the cars, score on the other half).

Run:  uv run python vehicle/lessons/06_zones_and_threshold.py      (needs lesson 05's cache)
"""

import importlib
import sys
sys.path.insert(0, "vehicle/lessons")
import numpy as np, cv2
from collections import defaultdict
l5 = importlib.import_module("05_link_cameras")
tracks = l5.load_tracks()
d = np.load(l5.CACHE)
keys = [tuple(k) for k in d["keys"]]; ids = [k[1] for k in keys]
mapping = l5.reid_to_gt(); _, gb = l5.lesson02.split_cars()
group_b = {mapping[c] for c in gb if c in mapping}; group_a = set(ids) - group_b
cands = l5.candidates(keys, tracks)
true_links = [(a, b) for a, b, _ in cands if ids[a] == ids[b]]
CELL = 128
def foot(box): return (int(box[3]) // CELL, int((box[0] + box[2]) / 2) // CELL)
def end_pt(k): fr = tracks[keys[k]]; return foot(fr[max(fr)])
def start_pt(k): fr = tracks[keys[k]]; return foot(fr[min(fr)])
# learn exit / entry zones per camera pair from group A true trips
exitz, entryz, gaps = defaultdict(set), defaultdict(set), defaultdict(list)
for a, b, g in cands:
    if ids[a] == ids[b] and ids[a] in group_a:
        pair = (keys[a][0], keys[b][0])
        exitz[pair].add(end_pt(a)); entryz[pair].add(start_pt(b)); gaps[pair].append(g)
def grow(cells, r=1):
    return {(y + dy, x + dx) for y, x in cells for dy in range(-r, r + 1) for dx in range(-r, r + 1)}
exitz = {p: grow(c) for p, c in exitz.items()}; entryz = {p: grow(c) for p, c in entryz.items()}
typ = {p: (np.median(v), np.percentile(v, 75) - np.percentile(v, 25)) for p, v in gaps.items()}
def allowed(a, b):
    p = (keys[a][0], keys[b][0])
    return end_pt(a) in exitz.get(p, ()) and start_pt(b) in entryz.get(p, ())
zc = [(a, b, g) for a, b, g in cands if allowed(a, b)]
tz = [(a, b) for a, b, _ in zc if ids[a] == ids[b]]
print(f"candidates {len(cands)} -> with zones {len(zc)}; true links kept {len(tz)}/{len(true_links)} "
      f"(group B kept {sum(ids[a] in group_b for a,b in tz)}/{sum(ids[a] in group_b for a,b in true_links)})")
v = d["v2"]
def unusual(a, b, g):
    med, s = typ.get((keys[a][0], keys[b][0]), (40, 30)); return max(0, abs(g - med) - s) / 60
def run(cs, w, label):
    score = lambda a, b, g: float(v[a] @ v[b]) - w * unusual(a, b, g)
    vals = np.arange(0.0, 0.9, 0.01)
    t = max(vals, key=lambda t: l5.score_links(l5.link(cs, score, t), true_links, ids, group_a)[2])
    p, r, f, _ = l5.score_links(l5.link(cs, score, t), true_links, ids, group_b)
    print(f"{label:<34} t={t:.2f}  P {p:.1%}  R {r:.1%}  F1 {f:.1%}")
run(cands, 0, "cosine only")
run(cands, 0.4, "+ travel time")
run(zc, 0, "+ zones")
run(zc, 0.4, "+ zones + travel time")
print("\ncosine of TRUE links: group A (trained on) vs group B (never seen)")
for name, grp in (("A", group_a), ("B", group_b)):
    s = [float(v[a] @ v[b]) for a, b in true_links if ids[a] in grp]
    print(f"  group {name}: median {np.median(s):.2f}, 10% lowest below {np.percentile(s, 10):.2f}")
s = [float(v[a] @ v[b]) for a, b, _ in zc if ids[a] != ids[b]]
print(f"  wrong pairs (after zones): median {np.median(s):.2f}, 99% below {np.percentile(s, 99):.2f}")
print("diagnosis only: F1 on group B for each threshold (zones + time):")
score = lambda a, b, g: float(v[a] @ v[b]) - 0.4 * unusual(a, b, g)
for t in (0.30, 0.35, 0.40, 0.45, 0.52):
    p, r, f, _ = l5.score_links(l5.link(zc, score, t), true_links, ids, group_b)
    print(f"  t={t:.2f}  P {p:.1%}  R {r:.1%}  F1 {f:.1%}")
print("\nhonest: 2-fold cross-validation inside group B (pick on one half, score on the other)")
bs = sorted(group_b); halves = [set(bs[0::2]), set(bs[1::2])]
def counts(links, grp):
    pred = [(a, b) for a, b in links if ids[a] in grp or ids[b] in grp]
    true = {(a, b) for a, b in true_links if ids[a] in grp}
    return sum((a, b) in true for a, b in pred), len(pred), len(true)
for cs, w, label in ((cands, 0, "cosine only"), (cands, 0.4, "+ travel time"), (zc, 0, "+ zones"), (zc, 0.4, "+ zones + travel time")):
    score = lambda a, b, g: float(v[a] @ v[b]) - w * unusual(a, b, g)
    good = made = possible = 0; ts = []
    for pick, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        t = max(np.arange(0.0, 0.9, 0.01), key=lambda t: l5.score_links(l5.link(cs, score, t), true_links, ids, pick)[2])
        g_, m_, p_ = counts(l5.link(cs, score, t), test); good += g_; made += m_; possible += p_; ts.append(round(t, 2))
    p, r = good / made, good / possible
    print(f"  {label:<24} thresholds {ts}  P {p:.1%}  R {r:.1%}  F1 {2*p*r/(p+r):.1%}")
