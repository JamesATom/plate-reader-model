# Vision models

One folder per model. Run every lesson from the project root, e.g.
`uv run python face/lessons/01_detect_faces.py face/data/samples/lena.jpg`.

| Folder | Task | Models used | Status |
|---|---|---|---|
| `plate/` | Plate reading (ANPR) | RF-DETR Nano detector + 2 fast-plate-ocr CCT readers, **trained by us** | done |
| `face/` | Face detection and recognition | YuNet detector + AdaFace IR-101 recognizer (SFace, EdgeFace compared), **pretrained, not retrained** | working app, tested on ChokePoint video |

## Where the models are

- **Plate:** [`plate/models/`](plate/models/) in this repo (Git LFS). Details, test results and how to
  reproduce them: [`plate/models/README.md`](plate/models/README.md).
  After cloning: `git lfs install && git lfs pull`.
- **Face:** made by other teams, so not copied into the repo. Download them from their official sources:
  `uv run python face/download_models.py` (checks SHA-256). Licenses are listed in that script;
  EdgeFace weights are non-commercial only.

## Setup

```bash
uv sync
git lfs pull
uv run python face/download_models.py
```

## Layout inside each model folder

- `lessons/` numbered step-by-step scripts
- `configs/` training configs
- `reports/` results and what we learned
- `models/` shared model files (plate only; face models are downloaded)
- `data/`, `runs/`, `output/` datasets, training runs, outputs (not in git)

The face app (enrol people, then recognise them in a video): see [`face/app/`](face/app/).
