# Plate models (trained by us)

Stored with **Git LFS**. After cloning run `git lfs pull` if the files below are only a few hundred bytes.

| File | What it does | Architecture | Trained on | Size |
|---|---|---|---|---|
| `plate_detector_rfdetr_nano.pth` | finds plates in a frame (**used**) | RF-DETR Nano, fine-tuned from COCO weights | 4,000 photos from Roboflow *Uzbek_Plates* (CC BY 4.0), 10 epochs | 121 MB |
| `plate_ocr_cct_s.keras` | reads the plate text (**used**, ensemble member 1) | fast-plate-ocr CCT, config `plate/configs/ocr/cct_s.yaml` | synthetic plates + real Roboflow crops + pseudo-labelled Roboflow crops (`run_ocr_v9.sh`) | 25 MB |
| `plate_ocr_cct_small.keras` | reads the plate text (**used**, ensemble member 2) | fast-plate-ocr CCT, config `plate/configs/ocr/cct_small.yaml` | same as above | 7 MB |
| `plate_detector_dfine_nano/` | alternative detector, only for the comparison in `plate/reports/` | D-FINE Nano (Hugging Face format) | same 4,000 photos | 15 MB |

Plate alphabet and size for both OCR models: `plate/configs/ocr/uz_plate_config.yaml`.
No photo of the teacher test set was used for training.

## Result on the teacher test set (plate_test_v1, 48 cameras, 422 plates)

Re-computed with exactly these files:

| Pipeline | conf | Recall | False boxes | Plate read exactly (end-to-end, 345 plates) |
|---|---|---|---|---|
| detector + OCR ensemble (lesson 32) | 0.72 | 90.8% | 18 | 86.1% (297) |
| + OCR also re-scores the boxes (lesson 34) | 0.5 | 92.2% | 25 | **87.0% (300)** |
| + OCR also re-scores the boxes (lesson 34) | 0.72 | 89.1% | 3 | 86.4% (298) |

Caveat: the confidence threshold, crop margins and the lesson 32/34 tricks were tuned while looking at
this same test set, so these numbers are somewhat optimistic. A new, unseen test set is needed for a fair score.

## Reproduce

```bash
T=plate/data/teacher_test/plate_test_v1
# 1. detect (640 px tiles) and merge boxes that overlap
uv run python plate/lessons/13_predict_folder.py rfdetr plate/models/plate_detector_rfdetr_nano.pth $T/images plate/output/det.csv 640
uv run python plate/lessons/15_merge_boxes.py plate/output/det.csv plate/output/det_merged.csv
# 2. read the plates with both OCR models; OCR confidence re-scores the boxes
MIN_CONF=0.02 SAVE_OCR=1 DEDUP=0 uv run python plate/lessons/32_read_better.py plate/output/det_merged.csv plate/output/ocrgate.csv \
    plate/models/plate_ocr_cct_s.keras plate/models/plate_ocr_cct_small.keras
uv run python plate/lessons/34_ocr_checks_detector.py plate/output/ocrgate.csv plate/output/predictions.csv 0.5
# 3. score
python3 $T/evaluate.py plate/output/predictions.csv --conf 0.5
```
