"""Enrol people: photos -> gallery file used by recognize.py.

Photo folder layout (either works):
    people/ali/1.jpg, people/ali/2.jpg, people/bobur/1.jpg   -> one sub-folder per person
    people/ali.jpg, people/bobur.jpg                          -> file name = person name

If a person has several photos, their embeddings are averaged.

Run:  uv run python face/app/enroll.py <photo folder> <gallery.npz> [--model AdaFace] [--skip id1,id2]
e.g.  uv run python face/app/enroll.py face/data/chokepoint/Neutral face/output/gallery_chokepoint.npz \
          --skip ID0023,ID0024,ID0025,ID0026,ID0027
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Detector, Gallery, Recognizer, align  # noqa: E402

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp"}

parser = argparse.ArgumentParser()
parser.add_argument("photos", type=Path)
parser.add_argument("gallery", type=Path)
parser.add_argument("--model", default="AdaFace", choices=["AdaFace", "SFace", "EdgeFace"])
parser.add_argument("--skip", default="", help="comma-separated names to leave out (to test 'unknown')")
args = parser.parse_args()
skip = {s for s in args.skip.split(",") if s}

detector, recognizer = Detector(), Recognizer(args.model)
faces_by_person = defaultdict(list)
for path in sorted(args.photos.rglob("*")):
    if path.suffix.lower() not in IMAGE_TYPES:
        continue
    name = path.parent.name if path.parent != args.photos else path.stem
    if name in skip:
        continue
    img = cv2.imread(str(path))
    if max(img.shape[:2]) > 1600:  # big phone/camera photos: the detector does not need that much
        s = 1600 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    faces = sorted(detector(img), key=lambda f: f.size, reverse=True)
    if not faces:
        print(f"  {path}: no face found, skipped")
        continue
    if len(faces) > 1:
        print(f"  {path}: {len(faces)} faces, using the biggest")
    faces_by_person[name].append(align(img, faces[0].points))

names, vectors = [], []
for name, faces in faces_by_person.items():
    v = recognizer(faces).mean(axis=0)
    names.append(name)
    vectors.append(v / np.linalg.norm(v))

args.gallery.parent.mkdir(parents=True, exist_ok=True)
Gallery(names, vectors, args.model).save(args.gallery)
print(f"enrolled {len(names)} people with {args.model} "
      f"({sum(len(f) for f in faces_by_person.values())} photos) -> {args.gallery}")
