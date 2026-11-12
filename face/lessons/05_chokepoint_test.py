"""Face lesson 05: real surveillance video. Who walked through the door?

Dataset: ChokePoint (NICTA, CC BY-NC 4.0, https://zenodo.org/records/815657)
  Neutral/IDxxxx.JPG          one clean photo per person            -> our "enrolment" database
  P1E_S1_C1..C3/*.jpg         3 ceiling cameras above a door, 800x600 -> "probe" video frames
  groundtruth/P1E_S1_Cx.xml   for every frame: who is there + where their eyes are

What happens:
  1. Enrol: every clean photo -> detect -> align -> embedding (one per person, per model).
  2. For every labelled face in the video: run OUR detector, keep the face whose eyes match
     the labelled eyes, align, embed, and find the closest enrolled person.
  3. Some people are removed from the database on purpose (UNKNOWN_IDS). For them the
     right answer is "unknown": their best score must stay BELOW the threshold.

Two ways to decide:
  per frame  every single video frame decides on its own
  per pass   all frames of one person walking past one camera are averaged first
             (this is what a real system does: it follows the person, then decides once)

Run:  uv run python face/lessons/05_chokepoint_test.py
"""

import importlib
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "face/lessons")
lesson02 = importlib.import_module("02_align_faces")
lesson04 = importlib.import_module("04_compare_models")  # MODELS: name -> embed(aligned face)

ROOT = Path("face/data/chokepoint")
SEQUENCE = "P1E_S1"
CAMERAS = ["C1", "C2", "C3"]
UNKNOWN_IDS = {"0023", "0024", "0025", "0026", "0027"}  # walk past the camera, but are NOT enrolled
CACHE = Path("face/output/face_05_embeddings.npz")


def read_groundtruth(xml_path):
    """-> list of (frame number, person id, eye midpoint x, y, eye distance)."""
    out = []
    for frame in ET.parse(xml_path).getroot().iter("frame"):
        for p in frame.iter("person"):
            l, r = p.find("leftEye"), p.find("rightEye")
            lx, ly, rx, ry = (float(l.get("x")), float(l.get("y")), float(r.get("x")), float(r.get("y")))
            out.append((frame.get("number"), p.get("id"), (lx + rx) / 2, (ly + ry) / 2, np.hypot(lx - rx, ly - ry)))
    return out


def matching_face(faces, cx, cy, eye_dist):
    """Our detection whose eyes are where the label says (within one eye-distance)."""
    best = None
    for box, points, conf in faces:
        mx, my = points[:2].mean(axis=0)
        d = np.hypot(mx - cx, my - cy)
        if d < max(eye_dist, 8) and (best is None or d < best[0]):
            best = (d, points)
    return None if best is None else best[1]


def build_embeddings():
    names = list(lesson04.MODELS)

    # 1. enrolment photos (4000x2672 -> shrink first, the detector does not need that much)
    gallery_ids, gallery = [], {m: [] for m in names}
    for path in sorted((ROOT / "Neutral").glob("ID*.JPG")):
        pid = re.search(r"ID(\d+)", path.name).group(1)
        if pid in UNKNOWN_IDS:
            continue
        img = cv2.imread(str(path))
        img = cv2.resize(img, None, fx=1200 / img.shape[0], fy=1200 / img.shape[0], interpolation=cv2.INTER_AREA)
        faces = lesson02.find_faces(img)
        if not faces:
            print(f"  enrol {path.name}: no face found")
            continue
        face = lesson02.align(img, faces[0][1])
        gallery_ids.append(pid)
        for m in names:
            gallery[m].append(lesson04.MODELS[m](face))
    print(f"enrolled {len(gallery_ids)} people (1 photo each); {len(UNKNOWN_IDS)} people kept out as 'unknown'")

    # 2. video frames
    probe_meta, probe, eye_dists, missed = [], {m: [] for m in names}, [], defaultdict(int)
    t0 = time.perf_counter()
    for cam in CAMERAS:
        labels = read_groundtruth(ROOT / "groundtruth" / f"{SEQUENCE}_{cam}.xml")
        by_frame = defaultdict(list)
        for row in labels:
            by_frame[row[0]].append(row)
        for frame_no, rows in by_frame.items():
            img = cv2.imread(str(ROOT / f"{SEQUENCE}_{cam}" / f"{frame_no}.jpg"))
            faces = lesson02.find_faces(img)
            for _, pid, cx, cy, eye_dist in rows:
                points = matching_face(faces, cx, cy, eye_dist)
                if points is None:
                    missed[cam] += 1
                    continue
                face = lesson02.align(img, points)
                probe_meta.append((cam, pid, frame_no))
                eye_dists.append(eye_dist)
                for m in names:
                    probe[m].append(lesson04.MODELS[m](face))
        print(f"  {cam}: {len(labels)} labelled faces, detector missed {missed[cam]}   ({time.perf_counter() - t0:.0f}s)")

    np.savez(CACHE, gallery_ids=np.array(gallery_ids), probe_meta=np.array(probe_meta), eye_dists=np.array(eye_dists),
             missed=np.array([missed[c] for c in CAMERAS]),
             **{f"gallery_{m}": np.array(gallery[m]) for m in names}, **{f"probe_{m}": np.array(probe[m]) for m in names})


def evaluate():
    d = np.load(CACHE)
    names = [k.removeprefix("gallery_") for k in d.files if k.startswith("gallery_") and k != "gallery_ids"]
    gallery_ids, meta, eye_dists = d["gallery_ids"], d["probe_meta"], d["eye_dists"]
    total = len(meta) + int(d["missed"].sum())
    print(f"\nvideo faces: {total} labelled, {len(meta)} found by our detector "
          f"({len(meta) / total:.0%}); eye distance median {np.median(eye_dists):.0f}px "
          f"(a face is roughly 2.5x that wide)")

    # passes: one person, one camera
    passes = defaultdict(list)
    for i, (cam, pid, _) in enumerate(meta):
        passes[(cam, pid)].append(i)

    print(f"\n{'model':<9} {'decide':<9} {'known: correct':>15} {'known score':>12} {'unknown score':>14} {'gap':>6}  {'best thr':>8}  {'errors at best thr':>18}")
    for m in names:
        G, P = d[f"gallery_{m}"], d[f"probe_{m}"]
        for mode in ["per frame", "per pass"]:
            if mode == "per frame":
                vecs, truth = P, [pid for _, pid, _ in meta]
            else:
                keys = list(passes)
                vecs = np.array([P[passes[k]].mean(axis=0) for k in keys])
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
                truth = [pid for _, pid in keys]
            scores = vecs @ G.T
            best = scores.argmax(axis=1)
            best_score = scores.max(axis=1)
            known = np.array([t not in UNKNOWN_IDS for t in truth])
            correct = np.array([gallery_ids[b] == t for b, t in zip(best, truth)])

            # a known person is handled right if the top match is them AND score >= threshold;
            # an unknown person is handled right if their best score < threshold
            def errors(thr):
                return int((known & ~(correct & (best_score >= thr))).sum() + (~known & (best_score >= thr)).sum())

            thr = min(np.unique(best_score), key=errors)
            k_scores = best_score[known & correct]
            u_scores = best_score[~known]
            print(f"{m:<9} {mode:<9} {correct[known].mean():>14.1%} "
                  f"{np.percentile(k_scores, 5):>12.2f} {u_scores.max():>14.2f} "
                  f"{np.percentile(k_scores, 5) - u_scores.max():>+6.2f}  {thr:>8.2f}  {errors(thr):>8}/{len(truth)}")
    print("\nknown score   = 5% of correct matches score lower than this")
    print("unknown score = highest score any not-enrolled person reached (should be below the threshold)")


if __name__ == "__main__":
    if not CACHE.exists() or "--rebuild" in sys.argv:
        build_embeddings()
    evaluate()
