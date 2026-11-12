"""Lesson 9: train RF-DETR Nano on a dataset folder.

Run:  uv run python plate/lessons/09_train_rfdetr.py plate/data/tiny_plates plate/runs/rfdetr_tiny 5
      uv run python plate/lessons/09_train_rfdetr.py plate/data/medium_plates plate/runs/rfdetr_medium 10
"""

import sys

from rfdetr import RFDETRNano

dataset_dir, output_dir, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])

# 1. Start from a model already trained on millions of everyday photos.
model = RFDETRNano()

# 2-8. The training loop: batches, guess, mistake score, turn knobs,
#      epochs, exam on valid, save the best file.
model.train(
    dataset_dir=dataset_dir,
    output_dir=output_dir,
    epochs=epochs,
    batch_size=4,      # 4 photos at a time
    num_workers=0,     # load photos in the main process (simplest on a Mac)
)
