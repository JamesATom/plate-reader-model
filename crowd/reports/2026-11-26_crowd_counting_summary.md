# Crowd counting on Tashkent video: results

**Date:** 2026-11-26
**Task:** Count the people in Tashkent metro, street and concert video (roadmap item 28: P2PNet, CLIP-EBC).

**What we did:**
- Tested both pretrained models on ShanghaiTech. Our scores match the authors' (P2PNet MAE 52.1 vs 51.96).
- Chose **P2PNet**: it marks every head, which allows zone counts, and it is 5-8 times faster than CLIP-EBC.
- Fine-tuned P2PNet on 19 labelled frames (276 people) from Tashkent YouTube videos.
- Scored it on **whole videos it never saw in training**.

## Results

Error = how many people off per frame (average). The frames have 0-37 people.

| | Authors' P2PNet | **Our fine-tuned P2PNet** |
|---|---|---|
| Test: metro report + Izzat Shukurov concert (7 frames, 155 people) | 48.6 (counted 453) | **5.4** (counted 181) |
| Validation: metro walk, street party, King Macarella concert (11 frames, 175 people) | 69.9 (counted 892) | **4.0** (counted 155) |
| Time per 1080p frame, Apple M4 GPU | 0.7 s | 0.9 s |

- **Fixed:** the pretrained model counted gravel, lamps and trains as people (149 on an empty frame). Ours gives 0.
- **Still wrong sometimes:**
  - It overcounts some close front rows (47 for 33).
  - It undercounts a dark night street (12 for 20).
  - It only counts heads it can see, so dark far stands at concerts are missed.

## What made the difference

1. **Training crops placed on heads.** Our frames are mostly empty floor and stage, so with random crops the model learned to count nobody.
2. **Counting at two image sizes** (full and half), merged. Big front-row faces were invisible to the model at full size. This alone cut the error from 7.8 to 4.7 on validation.
3. **An honest validation split.** All settings were chosen on validation videos; the test videos were used only for the final score.

## Conclusion

- **Ready for:** demos and trials. The counting app takes a video file, a camera stream (RTSP) or a webcam, and outputs a video with a dot on every head plus a CSV of the count over time.
- **Not yet:** exact counts. Expect a few people of error per frame (about 25%), and more on scenes unlike the training videos.
- **Next:** test on real fixed CCTV cameras, and label more dark-street and front-row frames.
- **License:** research use only (the weights go back to ShanghaiTech).

Full report: [2026-11-26_crowd_counting.md](2026-11-26_crowd_counting.md)
