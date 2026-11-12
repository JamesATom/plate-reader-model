"""Face recognition building blocks used by the app (same steps as lessons 01-04, in one place).

    detect  -> 5 points -> align to 112x112 -> embedding -> compare with the gallery
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lessons"))
from ir_resnet import load_adaface  # noqa: E402

MODELS_DIR = Path("face/models")
torch.set_grad_enabled(False)
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# ArcFace 112x112 template: right eye, left eye, nose, right mouth corner, left mouth corner
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


class Face:
    """One detected face in one frame."""

    def __init__(self, box, points, score):
        self.box = box          # x, y, w, h (ints)
        self.points = points    # 5x2 landmarks
        self.score = score      # detector confidence
        self.embedding = None   # filled later

    @property
    def size(self):
        return int(self.box[2])


class Detector:
    def __init__(self, min_confidence=0.7):
        self.net = cv2.FaceDetectorYN.create(str(MODELS_DIR / "yunet.onnx"), "", (320, 320),
                                             score_threshold=min_confidence)

    def __call__(self, img):
        h, w = img.shape[:2]
        self.net.setInputSize((w, h))
        _, raw = self.net.detect(img)
        if raw is None:
            return []
        return [Face(f[:4].astype(int), f[4:14].reshape(5, 2), float(f[14])) for f in raw]


def align(img, points):
    matrix, _ = cv2.estimateAffinePartial2D(points.astype(np.float32), TEMPLATE, method=cv2.LMEDS)
    return cv2.warpAffine(img, matrix, (112, 112))


class Recognizer:
    """aligned 112x112 BGR faces -> unit-length embeddings."""

    def __init__(self, name="AdaFace"):
        self.name = name
        if name == "SFace":
            self.sface = cv2.FaceRecognizerSF.create(str(MODELS_DIR / "sface.onnx"), "")
            return
        if name == "AdaFace":
            net = load_adaface(str(MODELS_DIR / "adaface_ir101/model.safetensors"))
        elif name == "EdgeFace":  # weights are CC BY-NC-SA: not for commercial use
            import timm
            net = timm.create_model("edgenext_base")
            net.reset_classifier(512)
            sd = torch.load(MODELS_DIR / "edgeface_base/edgeface_base.pt", map_location="cpu", weights_only=True)
            net.load_state_dict({k.removeprefix("model."): v for k, v in sd.items()})
        else:
            raise ValueError(f"unknown model {name}")
        self.net = net.eval().to(DEVICE)

    def __call__(self, faces_112):
        if not faces_112:
            return np.zeros((0, 128 if self.name == "SFace" else 512), np.float32)
        if self.name == "SFace":
            v = np.array([self.sface.feature(f).flatten() for f in faces_112], np.float32)
        else:
            rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in faces_112]).astype(np.float32) / 255.0
            x = torch.from_numpy((rgb - 0.5) / 0.5).permute(0, 3, 1, 2).to(DEVICE)
            v = self.net(x).cpu().numpy()
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class Gallery:
    """The enrolled people: one averaged embedding per person."""

    def __init__(self, names, vectors, model):
        self.names, self.vectors, self.model = list(names), np.asarray(vectors, np.float32), model

    def save(self, path):
        np.savez(path, names=np.array(self.names), vectors=self.vectors, model=self.model)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["names"].tolist(), d["vectors"], str(d["model"]))

    def best_match(self, embedding):
        """-> (name, similarity) of the closest enrolled person."""
        scores = self.vectors @ embedding
        i = int(scores.argmax())
        return self.names[i], float(scores[i])
