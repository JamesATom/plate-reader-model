"""Recognise people in a video: who walked past the camera, and when?

    every frame:  detect faces -> attach each face to a "track" (the same person in earlier frames)
                  -> good-quality faces get an embedding, added to that track's running average
    track ends:   (person not seen for a while) -> one final decision: enrolled name or UNKNOWN
                  -> one line in the events CSV

Why tracks: lesson 05 showed that deciding once per pass (averaging ~40 frames) removed the
mistakes that single frames make.

Input: a video file (.mp4, .avi, ...) or a folder of frames (.jpg), e.g. ChokePoint.
Output: <out>.mp4 with boxes and names, <out>.csv with one row per person pass.

Run:  uv run python face/app/recognize.py <video or frame folder> <gallery.npz> <out prefix>
e.g.  uv run python face/app/recognize.py face/data/chokepoint/P2E_S5_C1.1 \
          face/output/gallery_chokepoint.npz face/output/app_P2E_S5_C1
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Detector, Gallery, Recognizer, align  # noqa: E402

# Same person if the average face scores at least this much against the gallery.
# Chosen from the ChokePoint tests (lesson 05); MUST be re-checked on our own cameras.
DEFAULT_THRESHOLD = {"AdaFace": 0.45, "EdgeFace": 0.42, "SFace": 0.50}

MIN_FACE_PX = 40        # smaller faces are tracked but not used for identity (too little detail)
MAX_TURN = 0.35         # nose offset / eye distance: above this the head is turned too far (lesson 01)
MIN_GOOD_FACES = 3      # need at least this many good faces before naming anyone
MAX_MISSING = 15        # frames a track may be invisible before we call it finished


def frames_from(source):
    """Yield (frame index, BGR image) from a video file or a folder of images."""
    source = Path(source)
    if source.is_dir():
        for i, p in enumerate(sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".png"})):
            # frames named by number (00000226.jpg) keep that number: the folder may skip empty frames
            yield (int(p.stem) if p.stem.isdigit() else i), cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(source))
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield i, img
            i += 1


def head_turn(points):
    """0 = looking at the camera; bigger = turned sideways (nose far from the middle of the eyes)."""
    eye_mid = points[:2].mean(axis=0)
    eye_dist = np.linalg.norm(points[0] - points[1]) + 1e-6
    return abs(points[2][0] - eye_mid[0]) / eye_dist


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-6)


class Track:
    next_id = 1

    def __init__(self, face, frame):
        self.id = Track.next_id
        Track.next_id += 1
        self.box, self.first, self.last = face.box, frame, frame
        self.sum = None          # sum of good embeddings
        self.good = 0            # how many good faces were added
        self.seen = 1
        self.name, self.score = None, 0.0

    def add(self, face, frame):
        self.box, self.last = face.box, frame
        self.seen += 1

    def add_embedding(self, v):
        self.sum = v.copy() if self.sum is None else self.sum + v
        self.good += 1

    def decide(self, gallery, threshold):
        if self.good < MIN_GOOD_FACES:
            self.name, self.score = None, 0.0
            return
        mean = self.sum / np.linalg.norm(self.sum)
        name, score = gallery.best_match(mean)
        self.name, self.score = (name if score >= threshold else "UNKNOWN"), score


def match_faces_to_tracks(faces, tracks):
    """Greedy: biggest box overlap first. Returns {face index: track}."""
    pairs = sorted(((iou(f.box, t.box), i, t) for i, f in enumerate(faces) for t in tracks), key=lambda p: -p[0])
    used_f, used_t, out = set(), set(), {}
    for overlap, i, t in pairs:
        if overlap < 0.2:
            break
        if i in used_f or t.id in used_t:
            continue
        out[i] = t
        used_f.add(i)
        used_t.add(t.id)
    return out


def draw(img, track):
    x, y, w, h = track.box
    if track.name is None:
        color, label = (0, 200, 255), f"#{track.id} ..."
    elif track.name == "UNKNOWN":
        color, label = (0, 0, 255), f"#{track.id} UNKNOWN {track.score:.2f}"
    else:
        color, label = (0, 200, 0), f"#{track.id} {track.name} {track.score:.2f}"
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("gallery", type=Path)
    parser.add_argument("out", type=Path, help="output prefix: writes <out>.mp4 and <out>.csv")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--fps", type=float, default=30.0, help="frame rate (for times in the CSV)")
    args = parser.parse_args()

    gallery = Gallery.load(args.gallery)
    threshold = args.threshold or DEFAULT_THRESHOLD[gallery.model]
    detector, recognizer = Detector(), Recognizer(gallery.model)
    print(f"gallery: {len(gallery.names)} people, model {gallery.model}, threshold {threshold}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    events = []
    active, writer, t0 = [], None, time.perf_counter()

    def finish(track):
        track.decide(gallery, threshold)
        events.append({
            "track": track.id,
            "start_s": round(track.first / args.fps, 2),
            "end_s": round(track.last / args.fps, 2),
            "frames_seen": track.seen,
            "good_faces": track.good,
            "decision": track.name or "TOO_FEW_GOOD_FACES",
            "score": round(track.score, 3),
        })

    n, count = -1, 0
    for n, img in frames_from(args.source):
        count += 1
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(args.out.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"avc1"), args.fps, (w, h))

        faces = detector(img)
        matched = match_faces_to_tracks(faces, active)
        good_faces, good_tracks = [], []
        for i, face in enumerate(faces):
            track = matched.get(i)
            if track is None:
                track = Track(face, n)
                active.append(track)
            else:
                track.add(face, n)
            if face.size >= MIN_FACE_PX and head_turn(face.points) <= MAX_TURN:
                good_faces.append(align(img, face.points))
                good_tracks.append(track)
        for track, v in zip(good_tracks, recognizer(good_faces)):
            track.add_embedding(v)

        still_active = []
        for track in active:
            if n - track.last > MAX_MISSING:
                finish(track)
            else:
                still_active.append(track)
                if track.last == n:
                    track.decide(gallery, threshold)
                    draw(img, track)
        active = still_active
        writer.write(img)
        if count % 300 == 0:
            print(f"  frame {count}  ({time.perf_counter() - t0:.0f}s)")

    for track in active:
        finish(track)
    if writer is not None:
        writer.release()

    # drop very short tracks (a face seen for a few frames only: usually a false detection)
    events = [e for e in events if e["frames_seen"] >= 5]
    with open(args.out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track", "start_s", "end_s", "frames_seen", "good_faces", "decision", "score"])
        w.writeheader()
        w.writerows(events)

    print(f"\n{count} frames in {time.perf_counter() - t0:.0f}s -> {args.out.with_suffix('.mp4')}, {args.out.with_suffix('.csv')}")
    print(f"{'track':>5} {'time':>13} {'frames':>6} {'good':>5}  decision")
    for e in events:
        print(f"{e['track']:>5} {e['start_s']:>6.1f}-{e['end_s']:<6.1f} {e['frames_seen']:>6} {e['good_faces']:>5}  {e['decision']} {e['score']:.2f}")


if __name__ == "__main__":
    main()
