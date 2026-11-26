# Crowd counting on Tashkent video

**Date:** 2026-11-26
**Task:** Count the people in Tashkent metro, street and concert video (roadmap item 28: P2PNet, CLIP-EBC).

## Result

On videos it never saw in training, our fine-tuned P2PNet is **off by 4-5 people per frame**; the frames have 0-37 people. The authors' pretrained model is off by 49-70.

| Average error, people per frame | Authors' P2PNet | **Ours** |
|---|---|---|
| Test: metro report, Izzat Shukurov concert (7 frames, 155 people) | 48.6 | **5.4** |
| Validation: metro walk, street party, King Macarella concert (11 frames, 175 people) | 69.9 | **4.0** |

Speed: 0.9 s per 1080p frame on an Apple M4 GPU.

## What we did

1. **Checked both pretrained models on ShanghaiTech.** Our scores match the authors' (P2PNet MAE 52.1 vs 51.96; CLIP-EBC 6.6 vs 6.68).
2. **Chose P2PNet.** It marks every head, so zone counts are possible, and it is 5-8 times faster than CLIP-EBC.
3. **Tried it on Tashkent video.** It counted gravel, lamps and trains as people (149 on an empty frame) and missed big front-row faces.
4. **Labelled 19 frames from 7 YouTube videos** (276 people) and fine-tuned on them.
5. **Tested on whole videos that were never used for training.** All settings were chosen on the validation videos; the test videos gave only the final score.

## What made the difference

- **Training crops placed on heads.** Our frames are mostly empty floor and stage. With random crops the model learned to count nobody.
- **Counting at two image sizes** (full and half), with the points merged. At full size, big front-row faces were invisible to the model. This cut the validation error from 7.8 to 4.7.

## Limits

- **Visible heads only:** dark far stands at concerts, where people show only as phone lights, are not counted.
- **Weak spots:** it overcounts some close front rows (47 for 33) and undercounts a dark night street (12 for 20).
- **Small test:** 18 frames, all YouTube video. It has not been tested on real CCTV yet.
- **Who labelled:** the 4 metro test frames were labelled by hand. The rest were labelled with an AI assistant (Claude) and checked by eye.
- **License:** research use only, because the weights go back to ShanghaiTech.

## Next

1. Test on real fixed CCTV cameras.
2. Label more dark-street and front-row frames.
3. Grow the test set to 30+ frames.

**App:** `crowd/app/count_video.py` counts a video file, a camera stream or a webcam. It outputs a video with a dot on every head, plus a CSV of the count over time.
