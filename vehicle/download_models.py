"""Download the base vehicle re-ID model from its official source into vehicle/models/.

FastReID "SBS R50-ibn" trained on VeRi-776 (JDAI-CV/fast-reid, release v0.1.1). It was trained by
another team, so it is not copied into the repo; it is fetched from their release and checked with SHA-256.
Our fine-tuned models (vehicle/models/*.pt) start from it and are in the repo (Git LFS).

Licenses (check before any commercial use):
  FastReID code   Apache-2.0
  VeRi weights    trained on VeRi-776, which is released for research only = NON-commercial.
                  Our fine-tuned models inherit this.

Run:  uv run python vehicle/download_models.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "VeRi-SBS": ("vehicle/models/veri_sbs_R50-ibn.pth",
                 "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth",
                 "57fb9c17d88911ea64390bf5427f43511435e7f88f6eed9dbc969d4b611e53cd"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


for name, (path, url, expected) in MODELS.items():
    path = Path(path)
    if path.exists() and sha256(path) == expected:
        print(f"{name:<9} already there   {path}")
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{name:<9} downloading ...  {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "vehicle-models-download"})
    with urllib.request.urlopen(request) as response, open(path, "wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    if sha256(path) != expected:
        path.unlink()
        sys.exit(f"{name}: checksum mismatch, file deleted. The source may have changed.")
    print(f"{name:<9} ok               {path}")
