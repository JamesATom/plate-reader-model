# Crowd counting: P2PNet fine-tuned on Tashkent video

**Date:** 2026-11-26
**Task:** Crowd counting (roadmap item 28: P2PNet, CLIP-EBC; the feature built into Hikvision and Dahua cameras). Count the people in Tashkent metro, street and concert video.
**Status:** Working app. P2PNet fine-tuned by us; tested on videos it never saw in training.

## Summary

**On Tashkent video it has never seen, our model is off by about 4-5 people per frame, on frames of 0-37 people (about 25% of the average count). The authors' pretrained model is off by 49-70.**

| | Authors' P2PNet (ShanghaiTech) | **Our fine-tuned P2PNet** |
|---|---|---|
| Test: 7 frames, 155 people (metro report, Izzat Shukurov concert) | 48.6 per frame (counted 453) | **5.4 per frame** (counted 181) |
| Validation: 11 frames, 175 people (metro walk, street party, King Macarella concert) | 69.9 per frame (counted 892) | **4.0 per frame** (counted 155) |
| Speed, one 1920x1080 frame, Apple M4 GPU | 0.7 s | 0.9 s |

- **What fine-tuning fixed:** the pretrained model counts gravel, lamps and moving trains as heads, for example 149 "people" on an empty frame and 470 on a frame of 22. Ours gives 0 and 26.
- **What still goes wrong:**
  - It overcounts some close-up front rows (47 for 33, 20 for 12).
  - It undercounts a dark night street (11-12 for 17-20).
  - It only counts heads it can see: in a dark arena the far stands are just phone lights, so they are not counted.
- **How much to trust the numbers:** the test set is small (7 frames). Treat the result as "a few people off per frame", not as an exact figure.

## The two models we tried

Both are pretrained on ShanghaiTech (dense crowd photos). We first checked that our code reproduces the authors' published scores. MAE is the average error in people per image.

| ShanghaiTech test | P2PNet-A | CLIP-EBC-A | CLIP-EBC-B |
|---|---|---|---|
| Part A (182 dense images, 433 people on average): our MAE / authors' | 52.1 / 51.96 | 54.0 / 53.2 | 97.4 / - |
| Part B (316 street images, 124 people on average): our MAE / authors' | 21.2 / - | 13.7 / - | 6.6 / 6.68 |
| Time per image | **0.2 s** | 1.0-1.7 s | 1.0-1.7 s |

- **P2PNet** (Tencent Youtu, ICCV 2021): marks a point on every head (VGG16-bn + FPN).
- **CLIP-EBC** (ViT-B/16): estimates a density map. It is very good on Part B, but it is 5-8 times slower and gives no head positions.

**We chose P2PNet.** Head points make zone counts, line crossing and queue length possible, and it is the faster model. CLIP-EBC stays only as a second opinion.

## Why fine-tuning was needed

On Tashkent video the pretrained P2PNet failed in both directions:
- **False heads:** gravel, fairy lights, a motion-blurred train, and the text on a TV interview.
- **Missed people:** dense far crowds, and big near faces at concerts.

On the test frames below it counted 453 people where there are 155.

## Data

We downloaded YouTube videos of the Tashkent metro, streets and Humo Arena concerts. From them we picked frames and clicked every head with our labelling tool (`crowd/app/label_heads.py`).

- **Whole frames vs regions:** where a crowd was too dense to label completely, we labelled one box ("region") and the model is scored only inside that box.
- **Who labelled:** the 4 metro test frames were labelled by hand by me. The other frames were labelled with an AI assistant (Claude) and checked by eye.

| Set | Videos | Frames | People |
|---|---|---|---|
| Training | 7 (metro, street, New Year, Navruz, La Bouche and Jahongir Otajonov concerts) | 19 | 276 |
| Validation (chooses the training epoch and settings) | 3: metro walk, New Year street party, King Macarella concert | 11 | 175 |
| Test (never used for any choice) | 2: metro news report, Izzat Shukurov concert | 7 | 155 |

Validation and test are **whole videos** that are never used for training. Splitting frames of the same video would test the model on a place it has already seen.

## Training

We fine-tuned from the authors' ShanghaiTech-A weights, keeping their recipe:
- random 128x128 crops of each frame;
- one-to-one (Hungarian) matching of predicted points to labelled heads;
- AdamW;
- 150 epochs, about 6 minutes on the M4.

The script is `crowd/lessons/03_finetune_p2pnet.py`. Four things mattered:

1. **Crops placed on heads.**
   - **Problem:** ShanghaiTech photos are full of people, so any crop has heads. Our frames are mostly floor, ceiling and stage, so random crops were mostly empty. The model learned that "no head" is almost always right and collapsed to about 0 people per frame.
   - **Fix:** 3 of every 4 crops are now placed around a labelled head. The 4th stays random, so the model still learns that gravel and lamps are not heads.
2. **Wider zoom range during training** (0.3-1.5 instead of 0.7-1.3). One frame can hold heads from 5 to 150 pixels.
3. **An honest validation split.**
   - **Problem:** at first we kept the epoch with the best *test* score. That showed 5.4, but the same model's typical epochs were at 8-12, so the number was too kind.
   - **Fix:** now the epoch and every setting are chosen on validation videos only, and the test videos are looked at once at the end.
4. **Counting at two image sizes.**
   - **Problem:** P2PNet places each point within a few pixels of a fixed 8-pixel grid, and it learned heads of about 5-60 pixels. Faces of 80-150 pixels in the front rows were not found at all, and more training frames did not help.
   - **Fix:** we run the model on the full frame and again on the frame shrunk to half size, where big faces become normal-sized. Then we merge the points, counting a head found at both sizes once.
   - **Result:** validation error fell from 7.8 to 4.7 people per frame (test: from 6.7 to 5.4).
   - **Note:** we spotted this problem by looking at a test frame, but we adopted the fix only after the validation videos confirmed it.

**What did not help:** six more close-up concert frames from the Jahongir Otajonov concert. On a concert the model had never seen (King Macarella), they made no difference (3.2 vs 3.4 per frame). On the street party they made it worse. That model was dropped.

## Results per frame

Our model counts at two sizes; the authors' model makes one pass.

| Set | Video | Frame | True | Authors' | **Ours** |
|---|---|---|---|---|---|
| test | metro report | 72 s | 37 | 79 | **39** |
| test | metro report, blurred train | 246 s | 0 | 149 | **0** |
| test | metro report, interview | 291 s | 1 | 64 | **6** |
| test | metro report | 366 s | 33 | 91 | **47** |
| test | Izzat concert, front rows (region) | 126 s | 32 | 15 | **26** |
| test | Izzat concert (region) | 192 s | 19 | 15 | **20** |
| test | Izzat concert (region) | 210 s | 33 | 40 | **43** |
| val | metro walk | 6 s | 28 | 42 | **24** |
| val | metro walk | 159 s | 22 | 470 | **26** |
| val | metro walk | 228 s | 3 | 70 | **1** |
| val | New Year street party | 171 s | 22 | 57 | **18** |
| val | New Year street party | 801 s | 17 | 49 | **11** |
| val | New Year street party | 987 s | 20 | 159 | **12** |
| val | King Macarella concert (region) | 2112 s | 15 | 12 | **11** |
| val | King Macarella concert (region) | 2136 s | 15 | 10 | **14** |
| val | King Macarella concert (region) | 4744 s | 9 | 2 | **8** |
| val | King Macarella concert (region) | 6904 s | 12 | 20 | **20** |
| val | King Macarella concert (region) | 7256 s | 12 | 1 | **10** |

## The app

`crowd/app/count_video.py` counts a video file, an RTSP camera stream or a webcam:
- **Sampling:** every N seconds (`--every 2`).
- **Output video:** a dot on every head, the count, and a graph of the count over time.
- **Output table:** a CSV of the count over time.
- **Zones:** `--zone` also counts inside a box, such as a platform or the area in front of a stage.

It runs at about 1 frame per second at 1080p on an M4 MacBook. For example, it counted a 4-minute concert video every 2 seconds in under 2 minutes.

## Limits

- **Visible heads only:** people hidden in the dark (far arena stands lit only by phone lights) are not counted.
- **Small evaluation:** 18 labelled frames from 5 videos.
- **YouTube video, not CCTV:** fixed ceiling cameras look different (angle, lens, compression). The model has not been tested on real CCTV footage yet.
- **License:** research use only. The weights go back to ShanghaiTech, which is released for research.

## Next steps

1. **Test on real CCTV footage** from a fixed camera: a metro platform or a building entrance.
2. **Label more frames of the weak cases:** dark night streets and very close front rows.
3. **Grow the test set** to 30+ frames, so the error figure is more precise.

## Files

- **Model:** `crowd/models/p2pnet_tashkent.pt` (Git LFS); details in `crowd/models/README.md`.
- **Lessons:**
  - `crowd/lessons/01`: reproduces the ShanghaiTech scores.
  - `crowd/lessons/02`: the pretrained models on our video.
  - `crowd/lessons/03`: fine-tuning and scoring.
- **Tools:**
  - `crowd/app/find_crowd_frames.py`: finds frames worth labelling.
  - `crowd/app/label_heads.py`: labelling.
  - `crowd/app/count_video.py`: counting.
