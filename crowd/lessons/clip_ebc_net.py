"""CLIP-EBC (ViT-B/16): crowd counting by asking CLIP "how many people are in this small block?".
Plain PyTorch, inference only.

Re-written from Yiming-M/CLIP-EBC (MIT): models/clip/model.py, models/clip/_clip/{blocks,image_encoder,
text_encoder}.py, models/utils.py, utils/eval_utils.py. The original downloads all 9 OpenAI CLIP models
on import; here we build only ViT-B/16 and load the CLIP-EBC checkpoint, which already contains every weight.

How it counts:
    the image is cut into 224x224 windows (step 112, overlaps averaged)
    CLIP's image encoder (with 32 learned "visual prompt" tokens per layer) gives a 768-d vector
    per 16x16 patch -> upsampled to one vector per 8x8 block -> projected to CLIP's 512-d space
    CLIP's text encoder turns 5 sentences into 5 vectors:
        "There is no person." / "...one person." / "...two people." / "...three people." /
        "There are more than four people."
    each block picks how much it looks like each sentence (softmax) -> expected people in that block
    count = sum over all blocks ("density map")

Input: RGB image, any size (smaller than 224 is zero-padded). Output: density map (H/8, W/8).
"""

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PROMPTS = ["There is no person.", "There is one person.", "There are two people.",
           "There are three people.", "There are more than four people."]
# The value each sentence stands for. The last one is the average count of blocks with 4+ people in
# the training set, so it differs per training set (from the authors' configs/reduction_8.json).
ANCHORS = {"sha": [0, 1, 2, 3, 4.29992], "shb": [0, 1, 2, 3, 4.41009]}
WINDOW, STEP, REDUCTION = 224, 112, 8


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width, heads, causal_len=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1, self.ln_2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([("c_fc", nn.Linear(width, width * 4)), ("gelu", QuickGELU()),
                                              ("c_proj", nn.Linear(width * 4, width))]))
        # text encoder: each word only sees the words before it
        self.mask = None if causal_len is None else torch.full((causal_len, causal_len), float("-inf")).triu_(1)

    def forward(self, x):
        y = self.ln_1(x)
        mask = None if self.mask is None else self.mask.to(x.device, x.dtype)
        x = x + self.attn(y, y, y, need_weights=False, attn_mask=mask)[0]
        return x + self.mlp(self.ln_2(x))


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, causal_len=None):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, causal_len) for _ in range(layers)])


class ImageEncoder(nn.Module):
    """CLIP ViT-B/16 image tower, keeping every patch vector (not only the [CLS] summary)."""

    def __init__(self, width=768, layers=12, heads=12, patch=16, size=224):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, patch, stride=patch, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.zeros((size // patch) ** 2 + 1, width))
        self.ln_pre, self.ln_post = nn.LayerNorm(width), nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)


class TextEncoder(nn.Module):
    def __init__(self, width=512, layers=12, heads=8, context=77, vocab=49408, embed=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, width)
        self.positional_embedding = nn.Parameter(torch.zeros(context, width))
        self.transformer = Transformer(width, layers, heads, causal_len=context)
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.zeros(width, embed))

    def forward(self, tokens):
        x = (self.token_embedding(tokens) + self.positional_embedding).permute(1, 0, 2)
        x = self.ln_final(self.transformer.resblocks(x).permute(1, 0, 2))
        return x[torch.arange(len(x)), tokens.argmax(-1)] @ self.text_projection   # vector at end-of-text token


class BasicBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.conv2, self.bn2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)

    def forward(self, x):
        return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))


class ClipEBC(nn.Module):
    def __init__(self, anchors, num_vpt=32):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_decoder = nn.Sequential(BasicBlock(768))
        self.projection = nn.Conv2d(768, 512, 1)
        self.logit_scale = nn.Parameter(torch.zeros(()))
        self.num_vpt = num_vpt
        for i in range(12):   # "deep visual prompt tuning": 32 extra learned tokens fed into every layer
            setattr(self, f"vpt_{i}", nn.Parameter(torch.zeros(num_vpt, 768)))
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("tokens", tokenize(PROMPTS), persistent=False)
        self.text_features = None

    def encode_image(self, x):
        enc = self.image_encoder
        b, _, h, w = x.shape
        gh, gw = h // 16, w // 16
        x = enc.conv1(x).flatten(2).permute(0, 2, 1)                          # (B, patches, 768)
        x = torch.cat([enc.class_embedding.expand(b, 1, -1), x], 1)
        pos = enc.positional_embedding
        if (gh, gw) != (14, 14):                                              # stretch position codes
            grid = pos[1:].T.reshape(1, -1, 14, 14)
            grid = F.interpolate(grid, size=(gh, gw), mode="bicubic").reshape(-1, gh * gw).T
            pos = torch.cat([pos[:1], grid])
        x = enc.ln_pre(x + pos).permute(1, 0, 2)                              # (1 + patches, B, 768)
        for i, block in enumerate(enc.transformer.resblocks):
            vpt = getattr(self, f"vpt_{i}").unsqueeze(1).expand(-1, b, -1)
            x = block(torch.cat([x[:1], vpt, x[1:]]))
            x = torch.cat([x[:1], x[1 + self.num_vpt:]])                      # drop the prompt tokens again
        x = enc.ln_post(x.permute(1, 0, 2))[:, 1:]                            # patches only
        return x.permute(0, 2, 1).reshape(b, -1, gh, gw)

    def forward(self, x):
        """x: normalised (B, 3, H, W) -> expected people per 8x8 block, (B, 1, H/8, W/8)."""
        f = self.encode_image(x)
        f = F.interpolate(f, scale_factor=2, mode="bilinear")                # stride 16 -> 8
        f = F.normalize(self.projection(self.image_decoder(f)).permute(0, 2, 3, 1), dim=-1)
        if self.text_features is None:                                        # text never changes: do it once
            self.text_features = F.normalize(self.text_encoder(self.tokens), dim=-1)
        logits = self.logit_scale.exp() * f @ self.text_features.T            # (B, H, W, 5)
        probs = logits.permute(0, 3, 1, 2).softmax(1)
        return (probs * self.anchors).sum(1, keepdim=True)


def tokenize(texts, context=77):
    """CLIP's byte-pair tokens, padded with 0 to 77 like OpenAI's clip.tokenize."""
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    out = torch.zeros(len(texts), context, dtype=torch.long)
    for i, t in enumerate(texts):
        ids = tok(t)["input_ids"]                       # starts with <|startoftext|>, ends with <|endoftext|>
        out[i, :len(ids)] = torch.tensor(ids)
    return out


def load_clip_ebc(path, trained_on, device="cpu"):
    """trained_on: "sha" or "shb" (picks the matching value for the "more than four" sentence)."""
    net = ClipEBC(ANCHORS[trained_on])
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return net.to(device).eval()


@torch.no_grad()
def density_map(net, image_rgb, batch=64):
    """image_rgb: HxWx3 uint8. Returns the density map (H/8 x W/8); its sum is the count."""
    x = (image_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    h0, w0 = x.shape[:2]
    x = np.pad(x, ((0, max(0, WINDOW - h0)), (0, max(0, WINDOW - w0)), (0, 0)))
    h, w = x.shape[:2]
    x = torch.from_numpy(x).permute(2, 0, 1)[None].to(next(net.parameters()).device)

    ys = [min(i * STEP, h - WINDOW) for i in range(math.ceil((h - WINDOW) / STEP) + 1)]
    xs = [min(j * STEP, w - WINDOW) for j in range(math.ceil((w - WINDOW) / STEP) + 1)]
    spots = [(y, x0) for y in ys for x0 in xs]
    crops = torch.cat([x[..., y:y + WINDOW, x0:x0 + WINDOW] for y, x0 in spots])
    preds = torch.cat([net(crops[i:i + batch]) for i in range(0, len(crops), batch)]).cpu().numpy()

    total = np.zeros((h // REDUCTION, w // REDUCTION), np.float32)
    seen = np.zeros_like(total)
    k = WINDOW // REDUCTION
    for (y, x0), p in zip(spots, preds):
        total[y // REDUCTION:y // REDUCTION + k, x0 // REDUCTION:x0 // REDUCTION + k] += p[0]
        seen[y // REDUCTION:y // REDUCTION + k, x0 // REDUCTION:x0 // REDUCTION + k] += 1
    return (total / seen)[:math.ceil(h0 / REDUCTION), :math.ceil(w0 / REDUCTION)]   # drop the padding
