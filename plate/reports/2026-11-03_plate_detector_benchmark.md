# Plate detector benchmark: RF-DETR Nano vs D-FINE Nano

**Date:** 2026-11-03
**Goal:** Pick the plate *detector* (stage 1 of the ANPR pipeline: find where the plate is) for Uzbek license plates. Reading the characters (OCR) is a separate, later stage.
**Status:** First comparison on a small subset, trained on a Mac. This is a learning run, not a final decision.

## Summary

Both models found almost every plate. Accuracy was almost tied.

- **RF-DETR Nano** was more reliable: 100/100 plates found and 1 false box.
- **D-FINE Nano** was lighter and faster: an 8× smaller file, faster inference and faster training, with slightly tighter boxes.

The test set is too easy and too small to choose a winner yet.

## Results (test split, 100 photos, 100 plates)

| Metric | RF-DETR Nano | D-FINE Nano |
|---|---|---|
| Plates found (confidence ≥ 0.5) | **100/100** | 99/100 |
| False boxes (confidence ≥ 0.5) | **1** | 4 |
| mAP@50 | **1.000** | 0.979 |
| mAP@50:95 | 0.714 | **0.727** |
| Inference, ms per photo (Apple MPS, batch 1) | 30 | **24** |
| Weights file size | 115 MB | **14 MB** |
| Training time (10 epochs, 1,000 photos) | 43 min | **13 min** |

Both models were scored by the same script ([lessons/12_benchmark.py](../lessons/12_benchmark.py)) with `torchmetrics` `MeanAveragePrecision`, on the same photos. The test split was never used for training or for choosing the best checkpoint.

## Validation curve per epoch (mAP@50:95 on the valid split)

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| RF-DETR | 0.659 | 0.674 | 0.678 | 0.675 | 0.699 | 0.718 | **0.724** | 0.713 | 0.712 | 0.711 |
| D-FINE | 0.664 | 0.662 | 0.652 | 0.712 | **0.720** | 0.678 | 0.712 | 0.707 | 0.700 | 0.697 |

Both runs saved the best epoch, which is the one used for the test results above. mAP@50 on valid was about 1.0 for both from epoch 2 onwards.

D-FINE train loss: 6.7 → **38.5 (spike at epoch 2)** → 13.3 → … → 10.2. RF-DETR train loss went down smoothly from 3.9 to 2.8. The two losses use different formulas, so their values can't be compared with each other.

## Data

- **Source:** [Uzbek_Plates on Roboflow Universe](https://universe.roboflow.com/suxrobs-workspace/uzbek_plates), CC BY 4.0, COCO format, 1 class (`plate`), all photos 640×640
- **Full dataset:** 8,133 train photos (Roboflow-augmented copies of about 3,000 originals, including flips and 90° rotations) and 302 valid photos
- **Subset used here:** a random 1,000 train photos. The original valid split was shuffled and cut into 100 for **valid** (model selection) and a separate 100 for **test** (final scores)
- **What the photos look like (from a manual check of 12 random samples):** daytime, close-up, phone camera, mostly garages and inspection areas. There are no night shots, no distant plates, no fixed road-camera views, and no personalised plates such as `70 AHMAD` (these became legal in 2026)

## Training setup

| | RF-DETR Nano | D-FINE Nano |
|---|---|---|
| Pretrained weights | `rf-detr-nano.pth` (COCO) | `ustc-community/dfine-nano-coco` |
| Library | `rfdetr` 1.11.2, built-in `model.train()` | `transformers` 5.22.1, my own training loop |
| Input resolution | 384 (multi-scale per batch) | 640 |
| Epochs / batch size | 10 / 4 | 10 / 4 |
| Learning rate | 1e-4 (encoder 1.5e-4), weight decay 1e-4, step scheduler | 1e-4 constant, AdamW, weight decay 1e-4 |
| EMA weights | yes | no |
| Augmentation | built-in RF-DETR augmentations | **none** |
| Gradient clipping | library default | 0.1 |
| Hardware | Apple Silicon Mac (MPS), no NVIDIA GPU | same |

## Caveats: why this isn't a final decision

1. **The training setups weren't identical.** RF-DETR had augmentation, EMA and an LR schedule. My D-FINE loop had none of them. The loss spike suggests the D-FINE setup can be improved.
2. **The test is easy.** Test photos come from the same source and conditions as the training photos, and mAP@50 is close to 1.0 for both models, so the test can't separate them.
3. **The test is small.** With 100 photos, a difference of 1–3 plates may be noise.
4. **Out-of-distribution check:** I ran the RF-DETR model trained on 200 photos on one of our own photos. It found the large plate (85% confidence) but **missed a small, partly hidden plate**, even at 2% confidence. The dataset has almost no small or distant plates.
5. **The speeds are from a Mac**, not from the NVIDIA production target (the plan mentions DeepStream). The speed ranking may change after ONNX/TensorRT export.

## Questions to discuss

1. Should D-FINE be retrained with augmentation, EMA and a warmup/LR schedule before comparing again, so the comparison is fair?
2. Which test set would separate the models: night, distance, motion blur, angle, personalised plates?
3. How many real road-camera frames should we label before deciding, and how should the train/valid/test split work (for example, by camera or by day, to avoid near-duplicates)?
4. Should the Roboflow rotated and flipped copies stay in training? Mirrored plates won't happen on real cameras.
5. Is a Nano model enough, or should we also compare the Small/Medium versions once a GPU is available?

## How to reproduce

```bash
uv run python lessons/08_make_tiny_dataset.py data/medium_plates 1000 100
uv run python lessons/09_train_rfdetr.py data/medium_plates runs/rfdetr_medium 10
uv run python lessons/11_train_dfine.py data/medium_plates runs/dfine_medium 10
uv run python lessons/12_benchmark.py data/medium_plates/test
```
