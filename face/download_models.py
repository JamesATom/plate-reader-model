"""Download the face models from their official sources into face/models/.

These models were trained by other people, so we do not keep copies in the repo:
each one is fetched from where its authors publish it, then checked with SHA-256.

Run:  uv run python face/download_models.py            (all models)
      uv run python face/download_models.py AdaFace    (just one)

Licenses of the WEIGHTS (check before any commercial use):
  YuNet     MIT                              OpenCV Zoo (OpenCV's own HuggingFace copy, same file)
  SFace     Apache-2.0                       OpenCV Zoo (OpenCV's own HuggingFace copy, same file)
  AdaFace   code MIT; weights trained on WebFace12M (research dataset), no explicit weight license
  EdgeFace  CC BY-NC-SA 4.0 = NON-commercial only
  KP-RPE and the *_4M models (CVLFace, MSU): code MIT; weights trained on WebFace4M/12M,
            no explicit weight license (research use, like AdaFace above)

The *_4M models are only for lesson 06 (which part of KP-RPE helps?).
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "YuNet": ("face/models/yunet.onnx",
              "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
              "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "SFace": ("face/models/sface.onnx",
              "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
              "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
    "AdaFace": ("face/models/adaface_ir101/model.safetensors",
                "https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m/resolve/main/model.safetensors",
                "2ea535a43877bd3de8091903935c783ce335be66a9f8917fae9a7a18ae4bbf56"),
    "EdgeFace": ("face/models/edgeface_base/edgeface_base.pt",
                 "https://huggingface.co/Idiap/EdgeFace-Base/resolve/main/edgeface_base.pt",
                 "95861c09b22810136f43ec98845e7f09bfc3c43f5a804984a7bd2eac20abc30c"),
    "KP-RPE": ("face/models/kprpe_vit_base_webface12m/model.safetensors",
               "https://huggingface.co/minchul/cvlface_adaface_vit_base_kprpe_webface12m/resolve/main/model.safetensors",
               "99d16ed4aac0fdf0fcc82526b9b70703f3ec8c3041bf1bf44bd22751536e65db"),
    "AdaFace_4M": ("face/models/adaface_ir101_webface4m/model.safetensors",
                   "https://huggingface.co/minchul/cvlface_adaface_ir101_webface4m/resolve/main/model.safetensors",
                   "21adb6220e8799a0e658f16946df9649c7269f432fe9810a7b9c4ad1241080a8"),
    "ViT_4M": ("face/models/vit_base_webface4m/model.safetensors",
               "https://huggingface.co/minchul/cvlface_adaface_vit_base_webface4m/resolve/main/model.safetensors",
               "5fafd6b7d599a3ede5fac5bd1d01ad05e9e93e89b39b7687d4a3bc93ff2aebc0"),
    "KP-RPE_4M": ("face/models/kprpe_vit_base_webface4m/model.safetensors",
                  "https://huggingface.co/minchul/cvlface_adaface_vit_base_kprpe_webface4m/resolve/main/model.safetensors",
                  "3c6d37ea874c2f38ffc9a7f0e9247efc994c3fb5c12d044759ac294e19d127f7"),
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
        print(f"{name:<10} already there   {path}")
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{name:<10} downloading ...  {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "face-models-download"})
    with urllib.request.urlopen(request) as response, open(path, "wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    if sha256(path) != expected:
        path.unlink()
        sys.exit(f"{name}: checksum mismatch, file deleted. The source may have changed.")
    print(f"{name:<10} ok               {path}")
