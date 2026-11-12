"""Face lesson 02: straighten ("align") every face to the same 112x112 position.

Every recognizer (SFace, ArcFace, AdaFace, EdgeFace) was trained on faces where
the eyes, nose and mouth sit on fixed spots of a 112x112 square. That standard
layout is called the ArcFace template (TEMPLATE below). We move the 5 points the
detector found onto those 5 fixed spots: rotate + resize + shift in one step.

Run:  uv run python face/lessons/02_align_faces.py face/data/samples/*.jpg
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Where the 5 points must land inside a 112x112 face (right eye, left eye, nose,
# right mouth corner, left mouth corner). Every modern recognizer uses these numbers.
TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

_detector = cv2.FaceDetectorYN.create("face/models/yunet.onnx", "", (320, 320), score_threshold=0.7)


def find_faces(img):
    """Lesson 01: every face as (box, 5 points, confidence), biggest face first."""
    h, w = img.shape[:2]
    _detector.setInputSize((w, h))
    _, faces = _detector.detect(img)
    if faces is None:
        return []
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return [(f[:4].astype(int), f[4:14].reshape(5, 2), float(f[14])) for f in faces]


def align(img, points):
    """Rotate, resize and shift the face so its 5 points land on TEMPLATE. Returns 112x112."""
    # "Partial" affine = only rotation + one zoom + shift, so the face is never squashed.
    matrix, _ = cv2.estimateAffinePartial2D(points.astype(np.float32), TEMPLATE, method=cv2.LMEDS)
    return cv2.warpAffine(img, matrix, (112, 112))


def tilt_degrees(points):
    """How tilted the head is: angle of the line through the two eyes."""
    (rx, ry), (lx, ly) = points[0], points[1]
    return float(np.degrees(np.arctan2(ly - ry, lx - rx)))


if __name__ == "__main__":
    rows = []
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        faces = find_faces(img)
        if not faces:
            print(f"{Path(path).name}: no face")
            continue
        (x, y, w, h), points, conf = faces[0]  # biggest face only
        print(f"{Path(path).name}: tilt {tilt_degrees(points):+5.1f} deg  ->  aligned to 112x112")

        # left: the face as the camera saw it (box cut out, made square, with its 5 points)
        before = img.copy()
        for px, py in points:
            cv2.circle(before, (int(px), int(py)), max(2, w // 60), (0, 0, 255), -1)
        side = max(w, h)
        cx, cy = x + w // 2, y + h // 2
        y0, x0 = max(cy - side // 2, 0), max(cx - side // 2, 0)
        before = cv2.resize(before[y0:y0 + side, x0:x0 + side], (224, 224))

        # right: the aligned face, with the template spots where the points must land
        after = cv2.resize(align(img, points), (224, 224), interpolation=cv2.INTER_CUBIC)
        for tx, ty in TEMPLATE:
            cv2.circle(after, (int(tx * 2), int(ty * 2)), 4, (0, 255, 0), -1)

        label = np.full((30, 448 + 10, 3), 255, np.uint8)
        cv2.putText(label, Path(path).stem, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        rows.append(np.vstack([label, np.hstack([before, np.full((224, 10, 3), 255, np.uint8), after])]))

    out = "face/output/face_02_aligned.png"
    cv2.imwrite(out, np.vstack(rows))
    print(f"saved {out}   (left: camera view, right: aligned 112x112, green = template spots)")
