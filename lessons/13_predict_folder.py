"""Lesson 13: run a trained detector on every photo in a folder -> predictions.csv

Optional tiling: cut the big photo into overlapping squares, run the model on
each square, then put all boxes back on the big photo. Small plates stay big
enough for the model to see them.

Run:  uv run python lessons/13_predict_folder.py rfdetr runs/rfdetr_4000 data/teacher_test/plate_test_v1/images output/pred_rfdetr.csv
      uv run python lessons/13_predict_folder.py rfdetr runs/rfdetr_4000 data/teacher_test/plate_test_v1/images output/pred_rfdetr_tile640.csv 640
"""

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import Image
from torchvision.ops import nms

kind, run_dir, images_dir, out_csv = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
TILE = int(sys.argv[5]) if len(sys.argv) > 5 else 0      # 0 = no tiling, whole photo only
OVERLAP = 0.25          # neighbouring tiles share 25%, so a plate on a border is whole in one of them
device = "mps" if torch.backends.mps.is_available() else "cpu"
LOW = 0.01      # keep almost everything; the scoring script can filter by confidence later

if kind == "rfdetr":
    from rfdetr import RFDETRNano
    model = RFDETRNano(pretrain_weights=str(run_dir / "checkpoint_best_total.pth"))

    def detect(photo):
        found = model.predict(photo, threshold=LOW)
        return torch.tensor(found.xyxy, dtype=torch.float32).reshape(-1, 4), torch.tensor(found.confidence, dtype=torch.float32)
else:
    from transformers import AutoImageProcessor, DFineForObjectDetection
    processor = AutoImageProcessor.from_pretrained(run_dir / "best")
    model = DFineForObjectDetection.from_pretrained(run_dir / "best").to(device).eval()

    def detect(photo):
        with torch.no_grad():
            out = model(**processor(images=photo, return_tensors="pt").to(device))
        found = processor.post_process_object_detection(out, threshold=LOW, target_sizes=[photo.size[::-1]])[0]
        return found["boxes"].cpu(), found["scores"].cpu()


def tile_starts(length):
    """Where each tile begins along one side, e.g. 0, 480, 960 ... last one touches the edge."""
    if length <= TILE:
        return [0]
    step = int(TILE * (1 - OVERLAP))
    starts = list(range(0, length - TILE, step))
    return starts + [length - TILE]


def detect_tiled(photo):
    boxes, scores = detect(photo)                   # whole photo: big, close plates
    all_boxes, all_scores = [boxes], [scores]
    if TILE:
        for top in tile_starts(photo.height):
            for left in tile_starts(photo.width):
                boxes, scores = detect(photo.crop((left, top, left + TILE, top + TILE)))
                boxes[:, [0, 2]] += left        # tile coordinates -> big photo coordinates
                boxes[:, [1, 3]] += top
                all_boxes.append(boxes)
                all_scores.append(scores)
    boxes, scores = torch.cat(all_boxes), torch.cat(all_scores)
    # The same plate is often found in 2-3 overlapping tiles: keep only the most confident box.
    keep = nms(boxes, scores, iou_threshold=0.5)
    return boxes[keep].tolist(), scores[keep].tolist()


photos = sorted(images_dir.glob("*.jpg"))
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "x1", "y1", "x2", "y2", "conf", "text"])
    for i, path in enumerate(photos, 1):
        boxes, scores = detect_tiled(Image.open(path).convert("RGB"))
        for (x1, y1, x2, y2), conf in zip(boxes, scores):
            writer.writerow([path.name, round(x1), round(y1), round(x2), round(y2), round(conf, 4), ""])
        if i % 50 == 0:
            print(f"{i}/{len(photos)} photos", flush=True)
print(f"saved {out_csv}")
