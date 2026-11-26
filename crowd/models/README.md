# Crowd models

| File | What | Size |
|---|---|---|
| `p2pnet_tashkent.pt` | P2PNet (VGG16-bn + FPN, head points) fine-tuned by us on Tashkent metro and concert video frames | 43 MB (float16, Git LFS) |

The downloaded third-party weights (`p2pnet_shtech_a.pth`, `clip_ebc_*.pth`) live in the same folder
but are not in git: `uv run python crowd/download_models.py`.

## How to use

```bash
uv run python crowd/app/count_video.py VIDEO_OR_STREAM --every 2
```

In code: `net = load_p2pnet("crowd/models/p2pnet_tashkent.pt", device)` and
`points = count_heads(net, image_rgb)` (both in `crowd/lessons/p2pnet_net.py`). Use `count_heads`
(two image sizes merged), not a single `find_heads` pass: one pass misses big near faces.

## How it was made

Start: the authors' ShanghaiTech-A weights. Fine-tuned with `crowd/lessons/03_finetune_p2pnet.py`
on 19 labelled frames (276 heads) from 7 YouTube videos of the Tashkent metro, streets and concerts:

```bash
uv run python crowd/lessons/03_finetune_p2pnet.py --test j8yzZRSuPMk zLMhJwHz0fc \
    --val 8ifg0ozfQU4 V3ZuFIBkTB0 --scale 0.3 1.5 --epochs 150
```

The labelled frames themselves are not in the repo (they are frames of other people's videos).

## How good it is

Error = |counted - true| heads per frame, counted with `count_heads`. Validation and test videos were
never used for training; the saved epoch was chosen on validation only, so the test number is honest.

| Set | Videos | Frames / heads | Error per frame |
|---|---|---|---|
| Validation | metro walk, New Year street party, King Macarella concert (Humo Arena) | 11 / 175 | 4.0 |
| Test | metro news report, Izzat Shukurov concert (Humo Arena) | 7 / 155 | 5.4 |
| Test, authors' model before fine-tuning (one pass) | same | 7 / 155 | 49 (gravel, lamps and trains counted as heads) |

Known limits:
- It counts **visible heads**. In a dark arena the far stands are only phone lights, so a wide shot of
  a full arena counts the near rows only.
- It over-counts some close-up front rows (20 for 12) and under-counts dark night streets.
- The evaluation is small (18 frames); expect several heads of error per frame, more on new kinds of scenes.
- Labels: test metro frames by hand by us; the other frames were labelled by Claude and checked by eye.

## License

Research and non-commercial use only: the starting weights were trained on ShanghaiTech, which is
released for research. The P2PNet code is BSD-3.
