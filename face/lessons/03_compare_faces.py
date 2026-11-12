"""Face lesson 03: turn each aligned face into 128 numbers (an "embedding") and compare them.

The recognizer (SFace) was trained so that photos of the SAME person give number lists
pointing in nearly the same direction, and DIFFERENT people point in different directions.
We measure that with cosine similarity:
     1.0 = exactly the same direction (same person, same photo)
     0.0 = unrelated
SFace's authors recommend: similarity >= 0.363 means "same person".

(SFace gives 128 numbers. AdaFace/ArcFace give 512; the idea is identical.)

Run:  uv run python face/lessons/03_compare_faces.py face/data/samples/*.jpg
"""

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "face/lessons")
lesson02 = importlib.import_module("02_align_faces")  # find_faces, align

SAME_PERSON = 0.363  # threshold from the SFace authors; we will tune our own in a later lesson

recognizer = cv2.FaceRecognizerSF.create("face/models/sface.onnx", "")


def embed(aligned_face):
    """112x112 aligned face -> 128 numbers, scaled to length 1."""
    v = recognizer.feature(aligned_face).flatten()
    return v / np.linalg.norm(v)


def similarity(a, b):
    """Cosine similarity of two length-1 embeddings = their dot product."""
    return float(np.dot(a, b))


if __name__ == "__main__":
    names, vectors, thumbs = [], [], []
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        faces = lesson02.find_faces(img)
        if not faces:
            print(f"{Path(path).name}: no face, skipped")
            continue
        face = lesson02.align(img, faces[0][1])  # biggest face
        names.append(Path(path).stem)
        vectors.append(embed(face))
        thumbs.append(cv2.resize(face, (96, 96)))

    print("\nfirst 8 of the 128 numbers for each face:")
    for n, v in zip(names, vectors):
        print(f"  {n:<12} {np.round(v[:8], 2)}")

    # similarity of every face with every other face
    n = len(names)
    table = np.array([[similarity(vectors[i], vectors[j]) for j in range(n)] for i in range(n)])
    print(f"\nsimilarity table  (>= {SAME_PERSON} -> same person)")
    print(" " * 13 + "".join(f"{x[:10]:>11}" for x in names))
    for i in range(n):
        print(f"{names[i]:<13}" + "".join(f"{table[i, j]:>11.2f}" for j in range(n)))

    print("\nverdicts:")
    for i in range(n):
        for j in range(i + 1, n):
            s = table[i, j]
            verdict = "SAME person" if s >= SAME_PERSON else "different"
            print(f"  {names[i]:<12} vs {names[j]:<12} {s:5.2f}  ->  {verdict}")

    # picture: grid of faces with the score in every cell (green = same, red = different)
    cell = 96
    grid = np.full(((n + 1) * cell, (n + 1) * cell, 3), 255, np.uint8)
    for i in range(n):
        grid[(i + 1) * cell:(i + 2) * cell, :cell] = thumbs[i]
        grid[:cell, (i + 1) * cell:(i + 2) * cell] = thumbs[i]
        for j in range(n):
            s = table[i, j]
            y, x = (i + 1) * cell, (j + 1) * cell
            color = (200, 240, 200) if s >= SAME_PERSON else (210, 210, 245)
            grid[y + 2:y + cell - 2, x + 2:x + cell - 2] = color
            cv2.putText(grid, f"{s:.2f}", (x + 18, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    out = "face/output/face_03_similarity.png"
    cv2.imwrite(out, grid)
    print(f"\nsaved {out}")
