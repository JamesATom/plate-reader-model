"""Face lesson 01: find faces with a pretrained detector (YuNet). No training yet.

YuNet gives, for every face: a box, a confidence, and 5 landmarks
(right eye, left eye, nose tip, right mouth corner, left mouth corner).
The landmarks matter later: recognition needs the face "aligned" using the eyes.

Run:  uv run python face/lessons/01_detect_faces.py face/data/samples/lena.jpg
"""

import sys
from pathlib import Path

import cv2

photo = Path(sys.argv[1])
img = cv2.imread(str(photo))
h, w = img.shape[:2]

# score_threshold: drop faces the model is less than 70% sure about
detector = cv2.FaceDetectorYN.create("face/models/yunet.onnx", "", (w, h), score_threshold=0.7)
_, faces = detector.detect(img)
faces = [] if faces is None else faces

for f in faces:
    x, y, bw, bh = f[:4].astype(int)
    confidence = f[14]
    print(f"face {x} {y} {bw}x{bh}   confidence {confidence:.0%}")
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    for i in range(5):  # the 5 landmarks
        cv2.circle(img, (int(f[4 + 2 * i]), int(f[5 + 2 * i])), 3, (0, 0, 255), -1)
    cv2.putText(img, f"{confidence:.0%}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

print(f"found {len(faces)} faces")
out = f"face/output/face_01_{photo.stem}.png"
cv2.imwrite(out, img)
print(f"saved {out}")
