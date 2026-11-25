"""Download the crowd counting models from their official sources into crowd/models/.

These models were trained by other teams, so we do not keep copies in the repo:
each one is fetched from where its authors publish it, then checked with SHA-256.

  P2PNet-A     TencentYoutuResearch/CrowdCounting-P2PNet, weights/SHTechA.pth (in their git repo)
  CLIP-EBC-A   Yiming-M/CLIP-EBC release v1.0.0, ViT-B/16 "word" prompts, trained on ShanghaiTech A
  CLIP-EBC-B   same, trained on ShanghaiTech B
  CLIP-EBC weights come inside .tgz archives (~860 MB each, with training logs); we stream the archive
  and keep only best_mae.pth (~640 MB).

Run:  uv run python crowd/download_models.py               (all models)
      uv run python crowd/download_models.py CLIP-EBC-B    (just one)

Licenses (check before any commercial use):
  P2PNet code     BSD-3-Clause (Tencent)
  CLIP-EBC code   MIT
  ALL weights     trained on ShanghaiTech, a research dataset with no commercial licence.
                  CLIP-EBC also starts from OpenAI CLIP (MIT). Treat the weights as research only
                  until we retrain on data we are allowed to use.
"""

import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

CLIP_EBC = "https://github.com/Yiming-M/CLIP-EBC/releases/download/v1.0.0"
# name: (local path, url, file inside the archive or None, sha256 of the local file)
MODELS = {
    "P2PNet-A": ("crowd/models/p2pnet_shtech_a.pth",
                 "https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/raw/main/weights/SHTechA.pth",
                 None,
                 "506047732b128ff09efef18e94bfacbe35fcfef300e5e9eeeece259b0488c63f"),
    "CLIP-EBC-A": ("crowd/models/clip_ebc_vitb16_shtech_a.pth",
                   f"{CLIP_EBC}/ShanghaiTech_A_CLIP_EBC_ViT_B_16_Word.tgz",
                   "best_mae.pth",
                   "43823d35e7537fb1e197f7bed3d097329713bcabaf41b7771304b0733c310977"),
    "CLIP-EBC-B": ("crowd/models/clip_ebc_vitb16_shtech_b.pth",
                   f"{CLIP_EBC}/ShanghaiTech_B_CLIP_EBC_ViT_B_16_Word.tgz",
                   "best_mae.pth",
                   "c758780891bc2242083307ddcba2e240e74f288ca8f309dedd652c93ce624d2e"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy(src, path):
    with open(path, "wb") as out:
        while block := src.read(1 << 20):
            out.write(block)


wanted = sys.argv[1:] or list(MODELS)
for name in wanted:
    path, url, member, expected = MODELS[name]
    path = Path(path)
    if path.exists() and sha256(path) == expected:
        print(f"{name:<11} already there   {path}")
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{name:<11} downloading ...  {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "crowd-models-download"})
    with urllib.request.urlopen(request) as response:
        if member is None:
            copy(response, path)
        else:
            # read the archive as a stream and write out only the one file we want (never extract paths from it)
            with tarfile.open(fileobj=response, mode="r|gz") as tar:
                for entry in tar:
                    if entry.name == member and entry.isfile():
                        copy(tar.extractfile(entry), path)
                        break
                else:
                    sys.exit(f"{name}: {member} not found in the archive.")
    if sha256(path) != expected:
        path.unlink()
        sys.exit(f"{name}: checksum mismatch, file deleted. The source may have changed.")
    print(f"{name:<11} ok               {path}")
