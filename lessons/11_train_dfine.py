"""Lesson 11: train D-FINE Nano on the same dataset folder as RF-DETR.

RF-DETR hides the training loop inside model.train(). Here we write the
loop ourselves, so you can see every step from lesson "how training works".

Run:  uv run python lessons/11_train_dfine.py data/medium_plates runs/dfine_medium 10
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # run rare missing ops on CPU

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from transformers import AutoImageProcessor, DFineForObjectDetection

dataset_dir, output_dir, epochs = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
CHECKPOINT = "ustc-community/dfine-nano-coco"
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(1)

processor = AutoImageProcessor.from_pretrained(CHECKPOINT)


class CocoFolder(Dataset):
    """One split folder: photos + _annotations.coco.json."""

    def __init__(self, split):
        self.folder = dataset_dir / split
        coco = json.loads((self.folder / "_annotations.coco.json").read_text())
        self.images = coco["images"]
        self.boxes = {}
        for ann in coco["annotations"]:
            self.boxes.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        photo = Image.open(self.folder / info["file_name"]).convert("RGB")
        # Only one kind of object, so every box gets class 0 = "plate".
        anns = [{"bbox": a["bbox"], "category_id": 0, "area": a["area"], "iscrowd": 0}
                for a in self.boxes.get(info["id"], [])]
        return photo, {"image_id": info["id"], "annotations": anns}


def collate(batch):
    photos, targets = zip(*batch)
    inputs = processor(images=list(photos), annotations=list(targets), return_tensors="pt")
    inputs["sizes"] = [p.size[::-1] for p in photos]      # (height, width) for evaluation
    inputs["targets"] = targets
    return inputs


def evaluate(model, loader):
    """Exam: mAP on a split the model does not learn from."""
    model.eval()
    metric = MeanAveragePrecision(box_format="xywh")
    with torch.no_grad():
        for batch in loader:
            out = model(pixel_values=batch["pixel_values"].to(device))
            found = processor.post_process_object_detection(out, threshold=0.0, target_sizes=batch["sizes"])
            preds = [{k: v.cpu() for k, v in f.items()} for f in found]
            truth = [{"boxes": torch.tensor([a["bbox"] for a in t["annotations"]]).reshape(-1, 4),
                      "labels": torch.zeros(len(t["annotations"]), dtype=torch.long)}
                     for t in batch["targets"]]
            # predictions are x1y1x2y2 -> convert to xywh like the labels
            for p in preds:
                p["boxes"][:, 2:] -= p["boxes"][:, :2]
            metric.update(preds, truth)
    model.train()
    result = metric.compute()
    return float(result["map_50"]), float(result["map"])


# 1. Start from a model already trained on everyday photos (COCO).
#    Its old 80-class head is replaced by a new 1-class head: "plate".
model = DFineForObjectDetection.from_pretrained(
    CHECKPOINT, num_labels=1, id2label={0: "plate"}, label2id={"plate": 0},
    ignore_mismatched_sizes=True,
).to(device)

train_loader = DataLoader(CocoFolder("train"), batch_size=4, shuffle=True, collate_fn=collate)
valid_loader = DataLoader(CocoFolder("valid"), batch_size=4, collate_fn=collate)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

best = -1.0
output_dir.mkdir(parents=True, exist_ok=True)
log = open(output_dir / "metrics.csv", "w")
log.write("epoch,train_loss,val_mAP_50,val_mAP_50_95\n")

for epoch in range(epochs):                                   # 6. repeat for each epoch
    total = 0.0
    for batch in train_loader:                                # 2. a small group of photos
        labels = [{k: v.to(device) for k, v in l.items()} for l in batch["labels"]]
        out = model(pixel_values=batch["pixel_values"].to(device), labels=labels)  # 3. guess
        loss = out.loss                                       # 4. mistake score
        optimizer.zero_grad()
        loss.backward()                                       # 5. which way to turn each knob
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()                                      #    turn them a little
        total += loss.item()

    map50, map50_95 = evaluate(model, valid_loader)           # 7. exam on valid
    train_loss = total / len(train_loader)
    print(f"epoch {epoch + 1}/{epochs}  loss {train_loss:.3f}  mAP50 {map50:.3f}  mAP50:95 {map50_95:.3f}", flush=True)
    log.write(f"{epoch},{train_loss:.4f},{map50:.4f},{map50_95:.4f}\n")
    log.flush()

    if map50_95 > best:                                       # 8. keep the best version
        best = map50_95
        model.save_pretrained(output_dir / "best")
        processor.save_pretrained(output_dir / "best")
        print(f"  saved best ({best:.3f})", flush=True)
