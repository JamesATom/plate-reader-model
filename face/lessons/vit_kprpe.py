"""ViT face recognizer, with or without KP-RPE. Plain PyTorch, no training code.

KP-RPE = KeyPoint Relative Position Encoding (Kim, Su, Liu, Jain, Liu; MSU; CVPR 2024,
https://arxiv.org/abs/2403.14852). Trimmed copy of models/vit_kprpe from the CVLFace
repos on HuggingFace (minchul/cvlface_adaface_vit_base_kprpe_webface12m), keeping only
what inference needs, so it runs on the Mac without their C++/CUDA extension.

The idea in one paragraph:
  A ViT cuts the 112x112 face into 14x14 = 196 patches of 8x8 pixels. In every attention
  layer each patch decides how much to look at every other patch. A normal ViT learns
  that from the patch *grid* only ("the patch 3 to the right of me"), so it silently
  assumes the face sits exactly on the alignment template. KP-RPE also gives every patch
  its offset to the 5 face keypoints (eyes, nose, mouth corners) actually found in this
  crop, and turns those offsets into an extra attention bias. If the alignment is off by
  a few pixels, or the head is turned, the keypoints move and the attention moves with
  them. Same network, plus one Linear layer (keypoint_linear).
"""

import math

import torch
from torch import nn

# ---------- which pair of patches gets which "relative position bucket" ----------
# iRPE "product" method, ratio 1.9: offsets are bucketed per axis into -3..3
# (exact for |offset| <= 1, log-spaced further away), 7 x 7 = 49 buckets.
RATIO = 1.9
ALPHA, BETA, GAMMA = 1 * RATIO, 2 * RATIO, 8 * RATIO
BETA_INT = int(BETA)
NUM_BUCKETS = (2 * BETA_INT + 1) ** 2  # 49


def _piecewise_index(rel):
    """Offset in patches -> bucket in [-BETA_INT, BETA_INT]. Eq. 18 of the iRPE paper."""
    far = rel.abs() > ALPHA
    idx = rel.clone()
    r = rel[far].float()
    idx[far] = (torch.sign(r) * (ALPHA + torch.log(r.abs() / ALPHA) / math.log(GAMMA / ALPHA) * (BETA - ALPHA))
                .round().clip(max=BETA)).long()
    return idx


def bucket_ids(side):
    """(side*side, side*side) table: bucket of every (query patch, key patch) pair."""
    rows = torch.arange(side).view(side, 1).repeat(1, side)
    cols = torch.arange(side).view(1, side).repeat(side, 1)
    pos = torch.stack([rows, cols], 2).view(-1, 2)
    diff = pos[:, None, :] - pos[None, :, :]
    S = 2 * BETA_INT + 1
    return (_piecewise_index(diff[..., 0]) + BETA_INT) * S + (_piecewise_index(diff[..., 1]) + BETA_INT)


def rel_keypoints(keypoints, side):
    """keypoints (B, 5, 2) as x, y in 0..1 -> (B, side*side, 10): every patch centre minus every keypoint."""
    c = torch.linspace(0, 1, side + 1, device=keypoints.device, dtype=keypoints.dtype)
    c = (c[:-1] + c[1:]) / 2
    row, col = torch.meshgrid(c, c, indexing="ij")
    grid = torch.stack([col, row], -1).reshape(-1, 2)[None, :, None, :]  # (1, N, 1, 2) as x, y
    return (grid - keypoints[:, None, :, :]).flatten(2)


# ---------- the transformer ----------
class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, kp_bias=None, buckets=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B, heads, N, N)
        if kp_bias is not None:
            # KP-RPE: pair (i, j) gets kp_bias[b, h, i, bucket(i, j)], which depends on where
            # patch i sits relative to this face's keypoints. `buckets` is that lookup, flattened.
            attn = attn + kp_bias.flatten(2).index_select(2, buckets).view(B, self.num_heads, N, N)
        x = (attn.softmax(-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.ReLU6(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, kp_bias=None, buckets=None):
        x = x + self.attn(self.norm1(x), kp_bias, buckets)
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, patch, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class FaceViT(nn.Module):
    """112x112 RGB face (normalised to -1..1) [+ 5 keypoints] -> 512 numbers. CVLFace 'base' size."""

    def __init__(self, kprpe=True, img=112, patch=8, dim=512, depth=24, num_heads=16, mlp_ratio=3, out=512):
        super().__init__()
        self.side = img // patch
        n = self.side ** 2
        self.depth, self.num_heads = depth, num_heads
        self.patch_embed = PatchEmbed(patch, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, dim))
        self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.feature = nn.Sequential(nn.Linear(dim * n, dim, bias=False), nn.BatchNorm1d(dim, eps=2e-5),
                                     nn.Linear(dim, out, bias=False), nn.BatchNorm1d(out, eps=2e-5))
        # 10 offsets per patch -> one 49-bucket bias table per head per layer
        self.keypoint_linear = nn.Linear(10, NUM_BUCKETS * num_heads * depth) if kprpe else None
        # flat lookup: row i of kp_bias starts at i*49, then add the bucket of (i, j)
        flat = torch.arange(n)[:, None] * NUM_BUCKETS + bucket_ids(self.side)
        self.register_buffer("buckets", flat.flatten(), persistent=False)

    def forward(self, x, keypoints=None):
        B = x.shape[0]
        x = self.patch_embed(x) + self.pos_embed
        rel = None
        if self.keypoint_linear is not None:
            if keypoints is None:
                raise ValueError("KP-RPE model needs keypoints: (B, 5, 2), x and y in 0..1 of the 112x112 crop")
            rel = rel_keypoints(keypoints.to(x.dtype), self.side)  # (B, N, 10)
        per_layer = self.num_heads * NUM_BUCKETS
        for d, block in enumerate(self.blocks):
            bias = None
            if rel is not None:
                # rows d*784 .. (d+1)*784 of keypoint_linear belong to layer d. Computing one layer at a
                # time gives the same numbers as all at once, with 24x less memory.
                rows = slice(d * per_layer, (d + 1) * per_layer)
                bias = nn.functional.linear(rel, self.keypoint_linear.weight[rows], self.keypoint_linear.bias[rows])
                bias = bias.view(B, -1, self.num_heads, NUM_BUCKETS).transpose(1, 2)  # (B, heads, N, 49)
            x = block(x, bias, self.buckets)
        return self.feature(self.norm(x).reshape(B, -1))


def load_cvlface_vit(path, kprpe=True):
    """Load a CVLFace ViT-base model.safetensors (keys start with 'model.net.')."""
    from safetensors.torch import load_file
    weights = {k.removeprefix("model.net."): v for k, v in load_file(path).items()}
    net = FaceViT(kprpe=kprpe)
    net.load_state_dict(weights)
    return net.eval()


if __name__ == "__main__":
    net = load_cvlface_vit("face/models/kprpe_vit_base_webface12m/model.safetensors")
    print(net(torch.zeros(2, 3, 112, 112), torch.rand(2, 5, 2)).shape)
