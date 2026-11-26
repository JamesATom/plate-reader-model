"""P2PNet: crowd counting by finding one POINT per head. Plain PyTorch, inference only.

Re-written from TencentYoutuResearch/CrowdCounting-P2PNet (BSD-3-Clause): models/p2pnet.py,
models/backbone.py, util/misc.py. Only what is needed to turn an image into head points:

    VGG16-bn backbone (4 stages) -> small FPN (mixes coarse and fine features) -> stride-8 map
    every cell of that map has 4 "anchor" points; for each one the network predicts
        an offset (where the nearest head really is) and a score (is there a head at all?)
    count = number of points with score > 0.5

Input: RGB image, any size; it is zero-padded (after normalising) to a multiple of 128 like the
authors' test code. Output points are in the original image's pixels.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Backbone(nn.Module):
    """VGG16 with batch-norm, cut into 4 stages (output strides 2, 4, 8, 16)."""

    def __init__(self):
        super().__init__()
        f = list(torchvision.models.vgg16_bn(weights=None).features.children())
        self.body1 = nn.Sequential(*f[:13])
        self.body2 = nn.Sequential(*f[13:23])
        self.body3 = nn.Sequential(*f[23:33])
        self.body4 = nn.Sequential(*f[33:43])

    def forward(self, x):
        out = []
        for body in (self.body1, self.body2, self.body3, self.body4):
            x = body(x)
            out.append(x)
        return out


class Decoder(nn.Module):
    """Feature pyramid: upsample the deep (coarse) map and add it to the shallower ones."""

    def __init__(self, c3, c4, c5, size=256):
        super().__init__()
        self.P5_1, self.P5_2 = nn.Conv2d(c5, size, 1), nn.Conv2d(size, size, 3, padding=1)
        self.P4_1, self.P4_2 = nn.Conv2d(c4, size, 1), nn.Conv2d(size, size, 3, padding=1)
        self.P3_1, self.P3_2 = nn.Conv2d(c3, size, 1), nn.Conv2d(size, size, 3, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.P5_1(c5)
        p4 = self.P4_1(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        return self.P4_2(p4)  # stride 8: the only level P2PNet uses for its predictions


class Head(nn.Module):
    """conv-relu-conv-relu-conv. conv3/conv4 exist in the checkpoint but the original never uses them."""

    def __init__(self, out_per_anchor, anchors=4, size=256):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(size, size, 3, padding=1), nn.Conv2d(size, size, 3, padding=1)
        self.conv3, self.conv4 = nn.Conv2d(size, size, 3, padding=1), nn.Conv2d(size, size, 3, padding=1)
        self.output = nn.Conv2d(size, anchors * out_per_anchor, 3, padding=1)
        self.k = out_per_anchor

    def forward(self, x):
        x = self.output(F.relu(self.conv2(F.relu(self.conv1(x)))))
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.k)   # (B, cells * anchors, k)


def anchor_points(h, w, stride=8, row=2, line=2):
    """4 points per stride-8 cell (a 2x2 grid inside it), in the same order as the heads output them."""
    inner = (np.arange(row) + 0.5) * stride / row - stride / 2
    ax, ay = np.meshgrid(inner, inner)
    inner = np.stack([ax.ravel(), ay.ravel()], 1)                                      # (4, 2)
    cx, cy = np.meshgrid((np.arange((w + stride - 1) // stride) + 0.5) * stride,
                         (np.arange((h + stride - 1) // stride) + 0.5) * stride)
    centres = np.stack([cx.ravel(), cy.ravel()], 1)                                   # (cells, 2)
    return torch.from_numpy((centres[:, None] + inner[None]).reshape(-1, 2).astype(np.float32))


class P2PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.fpn = Decoder(256, 512, 512)
        self.regression = Head(2)        # (dx, dy) offset from the anchor point
        self.classification = Head(2)    # logits for (background, head)

    def forward(self, x):
        _, c3, c4, c5 = self.backbone(x)
        f = self.fpn(c3, c4, c5)
        points = self.regression(f) * 100 + anchor_points(*x.shape[2:]).to(x.device)
        return points, self.classification(f)    # logits (background, head); training needs them raw


def load_p2pnet(path, device="cpu"):
    """The authors' checkpoint ({"model": weights}) or ours from lesson 03 (the weights alone)."""
    net = P2PNet()
    state = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(state.get("model", state), strict=True)
    return net.to(device).eval()


@torch.no_grad()
def find_heads(net, image_rgb, threshold=0.5):
    """image_rgb: HxWx3 uint8. Returns (N, 2) head points (x, y) and their scores."""
    h, w = image_rgb.shape[:2]
    x = (image_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    H, W = -(-h // 128) * 128, -(-w // 128) * 128
    x = np.pad(x, ((0, H - h), (0, W - w), (0, 0)))
    x = torch.from_numpy(x).permute(2, 0, 1)[None].to(next(net.parameters()).device)
    points, logits = net(x)
    scores = logits.softmax(-1)[..., 1]
    keep = scores[0] > threshold
    return points[0][keep].cpu().numpy(), scores[0][keep].cpu().numpy()


def count_heads(net, image_rgb, threshold=0.4, sizes=(1.0, 0.5), merge=10):
    """find_heads at several image sizes, merged. Returns (N, 2) head points in full-size coordinates.

    P2PNet puts each point within a few pixels of a fixed 8 px grid and learned heads of ~5-60 px, so on
    1080p concert or metro video it misses big near faces (80-150 px) completely. Shrinking the frame
    to half makes them ordinary-sized, but makes far-away heads vanish - so we run both. A point from
    the smaller size is dropped if a point from a bigger size is already within merge / size pixels
    (same head, found twice). Chosen on the validation videos of lesson 03 (error 7.8 -> 4.7 heads per
    frame); threshold 0.4 because the fine-tuned model is a little shy.
    For FINE-TUNED weights only: the authors' ShanghaiTech model already over-counts our video (gravel,
    lamps), and a second pass doubles that (test error 49 -> 82). Score it with find_heads.
    """
    kept = np.zeros((0, 2), np.float32)
    for s in sizes:
        img = image_rgb if s == 1 else cv2.resize(image_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        p = find_heads(net, img, threshold)[0] / s
        if len(kept) and len(p):
            nearest = np.linalg.norm(p[:, None] - kept[None], axis=2).min(1)
            p = p[nearest > merge / s]
        kept = np.vstack([kept, p.astype(np.float32)])
    return kept
