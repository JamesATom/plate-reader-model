# KP-RPE (CVLFace) vs AdaFace: results

**Date:** 2026-11-24
**Task:** Test the speed and quality of KP-RPE (CVLFace, MSU, CVPR 2024) against AdaFace.

**Setup:**
- We used the authors' pretrained models and did not retrain them.
- Both models come from the same lab and were trained on the same data (WebFace12M) with the same loss (AdaFace). Only the network differs:
  - **AdaFace:** IR-101 CNN, 65 M parameters.
  - **KP-RPE:** ViT-Base that is also given the 5 face keypoints, 115 M parameters.

## Speed (Apple M4, GPU)

| | AdaFace | KP-RPE |
|---|---|---|
| GPU, 128 faces at once | **90 faces/s** | 49 faces/s |
| GPU, 1 face | **30.7 ms** | 35.4 ms |
| CPU, 1 face | **52 ms** | 70 ms |

## Quality

| Test | AdaFace | KP-RPE |
|---|---|---|
| LFW, easy photos | 99.80% | 99.82% |
| CPLFW, different head angles | 94.55% | **95.23%** |
| XQLFW, low-quality photos | 95.62% | **97.25%** |
| CPLFW, badly cropped faces | 91.63% | **94.63%** |
| CCTV video (ChokePoint), wrong answers out of 2905 frames | 23 | **1** |

**Setup check:** AdaFace's published scores are LFW 99.82% and CPLFW 94.57%. We got 99.80% and 94.55%.

## Conclusion

- **Accuracy:** KP-RPE is more accurate. The gap is largest on low-quality photos, badly cropped faces and CCTV video.
- **Cost:** KP-RPE is 1.15–1.8× slower and its model is 1.8× larger.
- **Where the gain comes from:** mostly from switching the network from a CNN to a ViT. The keypoints mainly help when a face is badly cropped.
- **Recommendation:** use KP-RPE when there are only a few faces per frame and a GPU. Keep AdaFace for many faces per frame or CPU-only machines.

Full report: [2026-11-24_kprpe_vs_adaface.md](2026-11-24_kprpe_vs_adaface.md)
