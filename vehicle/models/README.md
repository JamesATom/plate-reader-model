# Vehicle re-identification models

Re-ID = "is the car in camera B the same car we saw in camera A?" The model turns a car crop into
2048 numbers; the same car gives close numbers (cosine similarity), even when the plate is not readable.

## Files

| File | What it is | Where it comes from |
|---|---|---|
| `veri_sbs_R50-ibn.pth` | FastReID SBS R50-ibn, trained on VeRi-776 (the base model) | **not in git**: `uv run python vehicle/download_models.py` ([fast-reid release v0.1.1](https://github.com/JDAI-CV/fast-reid/releases/tag/v0.1.1)) |
| `roundabout_finetuned.pt` | base + fine-tuned on RoundaboutHD (lesson 02) | trained by us, Git LFS |
| `roundabout_uzbek_finetuned.pt` | lesson 02 model + fine-tuned on RoundaboutHD + Uzbek cars (lesson 04) | trained by us, Git LFS |

The `.pt` files are plain PyTorch state_dicts for the network in `vehicle/lessons/fastreid_net.py`:

```python
net = load_vehicle_reid("vehicle/models/veri_sbs_R50-ibn.pth")
net.load_state_dict(torch.load("vehicle/models/roundabout_uzbek_finetuned.pt", weights_only=True))
```

Input: RGB crop resized to 256x256, values 0..255. Output: 2048-d vector (normalise, then cosine).

## Results

Rank-1 = the closest crop is the right car. mAP = how high all crops of the right car are ranked.
All exam cars were never used for training (split by car, not by image).

| Model | RoundaboutHD Rank-1 | RoundaboutHD mAP | Uzbek Rank-1 | Uzbek Rank-5 | Uzbek mAP |
|---|---|---|---|---|---|
| VeRi base (no fine-tune) | 51.5% | 24.6% | 51.0% | 57.5% | 46.3% |
| + RoundaboutHD | **79.1%** | **56.9%** | 47.7% | 59.5% | 45.1% |
| + RoundaboutHD + Uzbek | 76.9% | 52.2% | **64.1%** | **65.4%** | **56.1%** |

- **RoundaboutHD exam:** 155 test cars (every 2nd car of the official test set), find the car in
  ANOTHER of the 4 cameras. 581 queries. On all 310 test cars the base model gets Rank-1 45.7%,
  mAP 19.7% (paper: 44.8 / 19.5).
- **Uzbek exam:** 17 cars, 153 queries, from YouTube videos of car bazaars / a street camera. One video
  is one camera, so the match must be more than 10 s away from the query (another moment / angle).
  Plates are covered, so the model cannot read them. Small exam: about ±4 points of noise.
- Fine-tuning on RoundaboutHD alone does **not** help on Uzbek cars: the model learns the place it is
  trained on. Uzbek data is needed for Uzbek cameras.
- Why not the paper's 99.7% on RoundaboutHD: its train set contains all 310 test cars (neighbouring
  frames), so that number measures memory, not re-ID.

## Data

- [RoundaboutHD](https://github.com/siri-rouser/RoundaboutHD) (University of Bath, MIT licence): 4 street cameras, 4K.
- Uzbek cars: mined by `vehicle/lessons/03_mine_videos.py` from YouTube videos, labelled automatically
  by our plate reader (same plate = same car), plates covered. 57 cars, 835 crops. **Not in git**:
  plates are personal data and the videos belong to their channels. Internal research use only.

## Licence

The base weights are trained on VeRi-776 (research only), so these fine-tuned models are
**non-commercial / research only** as well.

## Reproduce

```bash
uv run python vehicle/download_models.py
uv run python vehicle/lessons/01_reid_test.py            # base model on RoundaboutHD
uv run python vehicle/lessons/02_finetune.py --test      # lesson 02 model
uv run python vehicle/lessons/04_finetune_uzbek.py --test   # all three models, both exams
```
