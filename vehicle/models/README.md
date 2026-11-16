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

## Same car across the 4 cameras (lesson 05 and the app)

Cars never used in training (group B). Up to "App v2" the threshold was picked on group-A cars; from
lesson 06 on it is picked by 2-fold cross-validation inside group B (pick on half of the cars, score on the
other half), because the model has seen group-A cars: same-car similarity there is 0.84 (median) against
0.69 on new cars, so a group-A threshold is too strict and costs recall.

| Setup | Precision | Recall | F1 |
|---|---|---|---|
| Ground-truth tracks, time only (no re-ID) | 7.2% | 11.1% | 8.8% |
| Ground-truth tracks, VeRi base | 82.1% | 73.9% | 77.8% |
| Ground-truth tracks, + RoundaboutHD | 94.2% | 74.7% | 83.3% |
| Ground-truth tracks, + RoundaboutHD + Uzbek | 94.5% | 78.5% | 85.8% |
| Ground-truth tracks, + zones + travel time + cross-validated threshold (lesson 06) | **94.7%** | **96.2%** | **95.4%** |
| App v1: IoU tracker, + RoundaboutHD + Uzbek | 70.0% | 72.0% | 71.0% |
| App v2: ByteTrack-style tracker + road area | 78.7% | 75.6% | 77.1% |
| App v2 + travel time | 83.2% | 81.1% | 82.1% |
| App v3: + entry/exit zones + cross-validated threshold | 83.7% | 92.1% | 87.7% |
| **App v4: + duplicate-box removal** (current, cross-validated) | **84.8%** | **89.4%** | **87.0%** |
| App v4, checked by eye (see below) | **95.2-97.8%** | 90.2% | ~93% |

- Precision = share of the links we made that join the same car; recall = share of the real
  camera-to-camera trips we found (254 trips of 155 cars in 10 minutes).
- A busy roundabout: every true trip has ~135 candidate tracks in the next camera within 180 s, so travel
  time alone cannot do it.
- App v2: the tracker predicts each car's motion and lets weak boxes (half-hidden car) continue a track;
  cars outside the road area and parked cars are ignored; the link score is
  `cosine - 0.4 x (minutes the travel time is outside the usual range for that camera pair)`.
  Road areas, usual travel times, the 0.4 and the threshold 0.52 were all picked on group-A cars.
- The detector finds 99% of the car passages. What is left: a car's track still breaks into ~2 pieces
  (long hides behind the island), and 40 of the 59 wrong links go to vehicles the ground truth does not label.
- App v3: entry/exit zones (where cars leave camera a / enter camera b on each trip, learned from
  group-A cars) cut the candidate pairs 11x (499k -> 45k). Recall 81% -> 92%. Of the 48 wrong links in
  the last run, 31 go to vehicles the ground truth does not label.
- App v4: RF-DETR has no NMS inside and gave one car several boxes (on average ~26 overlapping pairs
  per frame), so one car got 2 tracks at the same time. Class-agnostic NMS before tracking: track
  pieces per car passage 2.02 -> 1.28 (1-piece passages 204 -> 500). F1 stays ~87-88%: the formal
  score is now limited by the ground truth itself.
- **The ground truth is incomplete.** All 36 links scored as wrong in the last run were checked by eye
  (crops of both sides): 23 are clearly the same car (the ground truth has no box for that passage, or
  gives the car another number), 7 are unclear, 6 are really wrong. Real precision is therefore
  260-267 of 273 = 95.2-97.8%. Missed trips (25): 11 removed by the zones, 8 because one side was already
  linked to another piece, 7 with too low similarity.
- Camera-bias removal (subtract each camera's mean vector) helps the base VeRi model only; not used.
- App: `uv run python vehicle/app/multicam.py` (RF-DETR Medium COCO detector, tracker, re-ID, links);
  `--tune` re-picks the threshold and travel-time weight by cross-validation.

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
