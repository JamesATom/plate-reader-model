"""Download the face models from their official sources into face/models/.

These models were trained by other people, so we do not keep copies in the repo:
each one is fetched from where its authors publish it, then checked with SHA-256.

Run:  uv run python face/download_models.py            (all models)
      uv run python face/download_models.py AdaFace    (just one)

Licenses of the WEIGHTS (check before any commercial use):
  YuNet     MIT                              OpenCV Zoo
  SFace     Apache-2.0                       OpenCV Zoo
  AdaFace   code MIT; weights trained on WebFace12M (research dataset), no explicit weight license
  EdgeFace  CC BY-NC-SA 4.0 = NON-commercial only
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "YuNet": ("face/models/yunet.onnx",
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
              "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "SFace": ("face/models/sface.onnx",
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
              "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
    "AdaFace": ("face/models/adaface_ir101/model.safetensors",
                "https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m/resolve/main/model.safetensors",
                "2ea535a43877bd3de8091903935c783ce335be66a9f8917fae9a7a18ae4bbf56"),
    "EdgeFace": ("face/models/edgeface_base/edgeface_base.pt",
                 "https://huggingface.co/Idiap/EdgeFace-Base/resolve/main/edgeface_base.pt",
                 "95861c09b22810136f43ec98845e7f09bfc3c43f5a804984a7bd2eac20abc30c"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


wanted = sys.argv[1:] or list(MODELS)
for name in wanted:
    path, url, expected = MODELS[name]
    path = Path(path)
    if path.exists() and sha256(path) == expected:
        print(f"{name:<9} already there   {path}")
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{name:<9} downloading ...  {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "face-models-download"})
    with urllib.request.urlopen(request) as response, open(path, "wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    if sha256(path) != expected:
        path.unlink()
        sys.exit(f"{name}: checksum mismatch, file deleted. The source may have changed.")
    print(f"{name:<9} ok               {path}")
