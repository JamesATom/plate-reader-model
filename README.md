# Vision models

One folder per model. Run every lesson from the project root, e.g.
`uv run python face/lessons/01_detect_faces.py face/data/samples/lena.jpg`.

| Folder | Task | Models used | Status |
|---|---|---|---|
| `plate/` | Plate reading (ANPR) | RF-DETR Nano detector + 2 fast-plate-ocr CCT readers, **trained by us** | done |
| `face/` | Face detection and recognition | YuNet detector + AdaFace IR-101 recognizer (SFace, EdgeFace compared), **pretrained, not retrained** | working app, tested on ChokePoint video |
| `vehicle/` | Vehicle re-identification (same car in another camera) | FastReID SBS R50-ibn (VeRi-776), **fine-tuned by us** on RoundaboutHD + Uzbek cars | multi-camera app working: same car linked across 4 cameras (F1 87-88% formal, precision ~95% checked by eye) |
| `crowd/` | Crowd counting (how many people in a frame) | P2PNet (head points) and CLIP-EBC ViT-B/16 (density map), pretrained on ShanghaiTech; **P2PNet fine-tuned on Tashkent metro/concert frames** | pretrained models verified on ShanghaiTech (match the authors' scores); P2PNet chosen. Pretrained it counts gravel/lights as heads (test error 49 heads per frame); fine-tuned on 19 labelled frames: about 7 heads per frame on unseen metro and concert videos, still undercounting close-up concert faces. More labelled frames next |

## Where the models are

- **Plate:** [`plate/models/`](plate/models/) in this repo (Git LFS). Details, test results and how to
  reproduce them: [`plate/models/README.md`](plate/models/README.md).
  After cloning: `git lfs install && git lfs pull`.
- **Face:** made by other teams, so not copied into the repo. Download them from their official sources:
  `uv run python face/download_models.py` (checks SHA-256). Licenses are listed in that script;
  EdgeFace weights are non-commercial only.
- **Vehicle:** our fine-tuned models are in [`vehicle/models/`](vehicle/models/) (Git LFS); results and
  how to reproduce: [`vehicle/models/README.md`](vehicle/models/README.md). The base model they start
  from is downloaded: `uv run python vehicle/download_models.py`. Non-commercial (VeRi-776).
- **Crowd:** made by other teams, downloaded from their official sources: `uv run python crowd/download_models.py`
  (checks SHA-256). Code is BSD-3 / MIT, but all weights are trained on ShanghaiTech (research only).

## Setup

```bash
uv sync
git lfs pull
uv run python face/download_models.py
uv run python vehicle/download_models.py
uv run python crowd/download_models.py
```

## Layout inside each model folder

- `lessons/` numbered step-by-step scripts
- `configs/` training configs
- `reports/` results and what we learned
- `models/` shared model files (plate, vehicle; face and base vehicle models are downloaded)
- `data/`, `runs/`, `output/` datasets, training runs, outputs (not in git)

The face app (enrol people, then recognise them in a video): see [`face/app/`](face/app/).
