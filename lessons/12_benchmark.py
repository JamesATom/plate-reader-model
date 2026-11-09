"""Lesson 12: benchmark - both models, same test photos, same scoring.

Run:  uv run python lessons/12_benchmark.py data/medium_plates/test runs/rfdetr_medium runs/dfine_medium
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import Image
from rfdetr import RFDETRNano
from torchmetrics.detection import MeanAveragePrecision
from transformers import AutoImageProcessor, DFineForObjectDetection

folder, rfdetr_run, dfine_run = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
coco = json.loads((folder / "_annotations.coco.json").read_text())
truth_boxes = {}
for ann in coco["annotations"]:
    x, y, w, h = ann["bbox"]
    truth_boxes.setdefault(ann["image_id"], []).append([x, y, x + w, y + h])
photos = [(Image.open(folder / img["file_name"]).convert("RGB"), img["id"]) for img in coco["images"]]
device = "mps" if torch.backends.mps.is_available() else "cpu"


# Each model gets the same job: photo in -> (boxes x1y1x2y2, scores) out.
rfdetr = RFDETRNano(pretrain_weights=str(rfdetr_run / "checkpoint_best_total.pth"))

def run_rfdetr(photo):
    found = rfdetr.predict(photo, threshold=0.01)
    return torch.tensor(found.xyxy).reshape(-1, 4), torch.tensor(found.confidence)


dfine_dir = dfine_run / "best"
dfine_proc = AutoImageProcessor.from_pretrained(dfine_dir)
dfine = DFineForObjectDetection.from_pretrained(dfine_dir).to(device).eval()

def run_dfine(photo):
    with torch.no_grad():
        inputs = dfine_proc(images=photo, return_tensors="pt").to(device)
        out = dfine(**inputs)
    found = dfine_proc.post_process_object_detection(out, threshold=0.01, target_sizes=[photo.size[::-1]])[0]
    return found["boxes"].cpu(), found["scores"].cpu()


def benchmark(name, run):
    metric = MeanAveragePrecision()
    found_plates = false_boxes = 0
    run(photos[0][0])                                   # warm-up, not timed
    start = time.perf_counter()
    for photo, image_id in photos:
        boxes, scores = run(photo)
        truth = torch.tensor(truth_boxes.get(image_id, [])).reshape(-1, 4)
        metric.update([{"boxes": boxes, "scores": scores, "labels": torch.zeros(len(boxes), dtype=torch.long)}],
                      [{"boxes": truth, "labels": torch.zeros(len(truth), dtype=torch.long)}])
        # Count like a real system would: only boxes the model is >= 50% sure about.
        sure = boxes[scores >= 0.5]
        found_plates += min(len(sure), len(truth))
        false_boxes += max(len(sure) - len(truth), 0)
    ms = (time.perf_counter() - start) / len(photos) * 1000
    m = metric.compute()
    return {"model": name, "found": f"{found_plates}/{sum(len(b) for b in truth_boxes.values())}",
            "false": false_boxes, "mAP50": float(m["map_50"]), "mAP50:95": float(m["map"]), "ms": ms}


rows = [benchmark("RF-DETR Nano", run_rfdetr), benchmark("D-FINE Nano", run_dfine)]
print(f"\n{'model':<14}{'found':>9}{'false':>7}{'mAP50':>8}{'mAP50:95':>10}{'ms/photo':>10}")
for r in rows:
    print(f"{r['model']:<14}{r['found']:>9}{r['false']:>7}{r['mAP50']:>8.3f}{r['mAP50:95']:>10.3f}{r['ms']:>10.0f}")
