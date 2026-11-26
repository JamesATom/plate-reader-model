# Face recognizer test: KP-RPE (CVLFace, MSU, CVPR 2024) vs AdaFace

**Date:** 2026-11-24
**Task:** Try KP-RPE from CVLFace for face recognition, then compare its speed and quality with AdaFace, the recognizer our face app uses now.
**Status:** Done. Pretrained models only; neither model was retrained.

## Summary

**KP-RPE is more accurate than AdaFace, and it is about 1.2-1.8× slower depending on how the faces are processed.**

- **Easy photos (LFW):** both models are near-perfect: 99.82% vs 99.80%.
- **Hard photos:** KP-RPE makes fewer mistakes.
  - Faces at different angles (CPLFW): 286 errors vs 327, out of 6000 pairs.
  - Low-quality photos (XQLFW): 165 errors vs 263.
- **Badly cropped faces:** KP-RPE barely drops, while AdaFace loses 2-3 points.
- **Real ceiling CCTV video (ChokePoint, indoor):** with each video frame judged on its own, KP-RPE made **1 error vs AdaFace's 23** (out of 2905 faces).
- **Cost:** KP-RPE is a 1.8× larger model.
  - Many faces at once on the GPU: 49 faces/s vs 90.
  - One face at a time on the GPU: 35 ms vs 31.
  - One face at a time on the CPU: 70 ms vs 52.

Part of KP-RPE's gain comes from its network type: a transformer (ViT) instead of a CNN. The other part comes from the keypoints themselves; see "Where the gain comes from" below.

## The two models

Both come from the same lab (Michigan State University, CVLFace project by Minchul Kim and Xiaoming Liu). Both were trained on the same data (WebFace12M) with the same loss (AdaFace). So the comparison isolates the network design.

| | AdaFace | KP-RPE |
|---|---|---|
| Network | IR-101 (CNN) | ViT-Base + KeyPoint Relative Position Encoding |
| Input | aligned 112×112 face | aligned 112×112 face + its 5 keypoints (eyes, nose, mouth corners) |
| Weights | `minchul/cvlface_adaface_ir101_webface12m` | `minchul/cvlface_adaface_vit_base_kprpe_webface12m` |
| Parameters / file | 65 M / 261 MB | 115 M / 460 MB |

**What KP-RPE adds:** a normal recognizer assumes the face sits exactly on a fixed template. KP-RPE is also told where the eyes, nose and mouth actually are in the crop. It uses those positions to steer its attention, so a crop that is a few pixels off, rotated, or of a turned head hurts it less.

**Where the keypoints come from:** our face detector (YuNet) already finds these 5 points for every face, so in our app the keypoints cost nothing extra.

**How we ran it:**
- The official CVLFace loader tries to compile a C++ extension and then stops the program, which does not work on a Mac.
- So we wrote a trimmed copy of the network: [vit_kprpe.py](../lessons/vit_kprpe.py).
- We checked it against the official CVLFace code, with the same weights and the same input. The outputs are **identical** (largest difference 0.0).

## Speed

Apple M4 (MacBook), PyTorch on the Apple GPU (MPS). No other program was using the GPU. Each number is the median of repeated runs after a warm-up run.

| | AdaFace | KP-RPE | KP-RPE is |
|---|---|---|---|
| GPU, 128 faces at once | **90 faces/s** | 49 faces/s | 1.8× slower |
| GPU, 1 face | **30.7 ms** | 35.4 ms | 1.15× slower |
| CPU, 1 face | **52 ms** | 70 ms | 1.34× slower |
| Model file | **261 MB** | 460 MB | 1.8× bigger |

Run: `uv run python face/lessons/06_kprpe_vs_adaface.py speed`

## Quality 1: standard benchmarks

Each benchmark has 6000 pairs of aligned 112×112 faces: 3000 pairs of the same person and 3000 of different people. Score = standard 10-fold accuracy (a threshold chosen on 9 folds, tested on the 10th). In brackets: wrong pairs out of 6000.

| Benchmark | What makes it hard | AdaFace | KP-RPE |
|---|---|---|---|
| LFW | nothing (easy, celebrity photos) | 99.80% (12) | **99.82% (11)** |
| CPLFW | the same person seen from very different angles | 94.55% (327) | **95.23% (286)** |
| XQLFW | one photo of each pair is blurred / low resolution | 95.62% (263) | **97.25% (165)** |

**Check that our setup is right:** AdaFace's published scores for this exact model are LFW 99.82% and CPLFW 94.57%. We measured 99.80% and 94.55%.

Data: aligned pairs from HuggingFace `LSIbabnikz/{lfw,cplfw,xqlfw}_bin`.

## Quality 2: badly aligned faces

These are the same pairs, but one face of every pair was shifted, rotated and zoomed at random, as if the detector had cropped it badly:
- **strong:** up to 12 px shift, 15° rotation and 15% zoom
- **extreme:** up to 16 px shift, 20° rotation and 20% zoom

KP-RPE is given the keypoints at their new positions. Every model saw the same random changes. Example faces: `face/output/face_06_misaligned.png`.

| Benchmark | AdaFace: normal → extreme | KP-RPE: normal → extreme |
|---|---|---|
| LFW | 99.80 → 99.52% | 99.82 → **99.82%** |
| CPLFW | 94.55 → 91.63% (**−2.9**) | 95.23 → **94.63%** (−0.6) |
| XQLFW | 95.62 → 93.72% (**−1.9**) | 97.25 → **96.68%** (−0.6) |

## Quality 3: real CCTV video (ChokePoint)

This is the same test as lesson 05.
- **Cameras:** 3 ceiling cameras above a door, 800×600.
- **Face size:** faces are small (median distance between the eyes 28-33 px).
- **Enrolment:** one clean photo per person, 24 people.
- **Unknown people:** 5 more people walk past who are *not* enrolled; the right answer for them is "unknown".
- **Counting errors:** a wrong name, a known person rejected, or an unknown person accepted.

| Video | Decision | AdaFace errors | KP-RPE errors |
|---|---|---|---|
| Indoor (P1E_S1), threshold tuned on this video | per frame (2905 faces) | 23 | **1** |
| | per pass (75 walks past a camera) | 0 | 0 |
| Outdoor, another day (P2E_S1), threshold **from the indoor video** | per frame (3677 faces) | 4 | **2** |
| | per pass (73 walks) | 3 | **1** |

**Safety margin between known and unknown people:** this is the score gap between the weakest correct match and the strongest unknown person, per frame, indoor. KP-RPE: **+0.19**. AdaFace: +0.08.

A larger margin means it is easier to set one threshold that works on a new camera.

## Where the gain comes from

We added extra models from the same authors to separate the two possible causes. All of these were trained on the smaller WebFace4M set.

| Model | Network | LFW | CPLFW | XQLFW | CPLFW extreme misalignment |
|---|---|---|---|---|---|
| IR-101, WebFace4M | CNN (like AdaFace) | 99.83% | 94.33% | 95.70% | 91.37% |
| ViT, WebFace4M | ViT, **no** keypoints | 99.83% | **95.00%** | **96.78%** | 94.23% |
| KP-RPE, WebFace4M | ViT **+** keypoints | 99.83% | 94.83% | 96.57% | **94.40%** |

We also ran KP-RPE (WebFace12M) with the true keypoints **hidden**, i.e. always told "the face is on the template":

| | CPLFW | CPLFW extreme | XQLFW extreme |
|---|---|---|---|
| KP-RPE, real keypoints | **95.23%** | **94.63%** | **96.68%** |
| KP-RPE, keypoints hidden | 94.18% | 91.27% | 95.40% |

**What this shows:**
1. **Most of the gain over AdaFace comes from switching CNN → ViT.** A plain ViT trained without keypoints is already better than the CNN on hard and misaligned faces.
2. **The KP-RPE network really does use the keypoints.** Hide them and it falls to the CNN's level on misaligned faces (91.27% vs AdaFace's 91.63% at CPLFW extreme).
3. **On the smaller WebFace4M training set, KP-RPE was not better than a plain ViT.** It was 0.2 points lower on normal hard pairs and 0.2 higher on misaligned ones. The KP-RPE paper reports a larger benefit; we did not see it on these benchmarks with these models.

## Keypoints: our detector vs CVLFace's own

CVLFace's official pipeline finds the 5 keypoints with its own landmark net (DFA). We use YuNet, our app's detector. We compared both on the same KP-RPE model:

| | LFW | CPLFW | XQLFW |
|---|---|---|---|
| Typical distance YuNet ↔ DFA keypoints | 2.7 px | 3.5 px | 2.9 px |
| KP-RPE with YuNet keypoints | 99.82% | 95.23% | 97.25% |
| KP-RPE with DFA keypoints | 99.80% | 95.57% | 97.13% |

**Result:** our detector is fine for KP-RPE.

YuNet found no face on 622 of 12,000 CPLFW faces (mostly side views) and on 288 of 12,000 XQLFW faces. For those we used the template instead. DFA helps slightly on such side views (+0.3 points).

## Recommendation for our face app

**Switch to KP-RPE if the app's frame rate allows it.**
- On the CCTV video it cut single-frame mistakes from 23 to 1.
- It kept a clearly safer margin between known and unknown people.
- At a typical door camera with a few faces per frame, 35 ms per face on the GPU is affordable.

**Keep AdaFace** where speed matters more than the last few points of accuracy:
- many faces per frame
- CPU-only machines

## Limits of this test

- **Speed** was measured on one Mac (M4, Apple GPU). An NVIDIA GPU will give different absolute numbers; the ratio between the models is the useful part.
- **XQLFW folds:** the XQLFW copy we used does not store its fold numbers. We used the official order (300 same + 300 different pairs per fold), as for LFW and CPLFW.
- **ChokePoint** has only 24 enrolled people and 5 unknown people, so the error counts there are small.
- **Licences:** both models were trained on WebFace4M/12M, which are research datasets with no explicit licence for the weights. Check before any commercial use.

## How to reproduce

```bash
uv run python face/download_models.py                       # models, SHA-256 checked
uv run python face/lessons/06_kprpe_vs_adaface.py speed     # speed (close other GPU programs)
uv run python face/lessons/06_kprpe_vs_adaface.py bench     # LFW / CPLFW / XQLFW + misalignment (~1.5 h on an M4)
uv run python face/lessons/06_kprpe_vs_adaface.py P1E_S1    # ChokePoint indoor
uv run python face/lessons/06_kprpe_vs_adaface.py P2E_S1    # ChokePoint outdoor, thresholds from indoor
```

- **Benchmark data:** `face/data/bench/` (HuggingFace `LSIbabnikz/*_bin`).
- **ChokePoint data:** `face/data/chokepoint/` (Zenodo record 815657).
- **Caches:** embeddings are cached in `face/output/face_06_bench/`, so a stopped run continues where it left off.
