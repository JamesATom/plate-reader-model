"""FastReID "SBS R50-ibn": the vehicle re-ID network trained on VeRi-776. Plain PyTorch, inference only.

Re-written from JDAI-CV/fast-reid (Apache-2.0): fastreid/modeling/backbones/resnet.py,
fastreid/layers/{batch_norm,non_local,pooling}.py, fastreid/modeling/heads/embedding_head.py.
Only what is needed to turn a car crop into 2048 numbers:

    ResNet-50 with IBN (half InstanceNorm, half BatchNorm in layer1-3: less sensitive to colour/light)
    + Non-local blocks (each spot of the image can look at every other spot)
    + GeM pooling (between average and max pooling) + BatchNorm "neck"

Input: RGB image resized to 256x256, values 0..255 (the network normalises inside).
"""

import torch
import torch.nn.functional as F
from torch import nn


class IBN(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.half = planes // 2
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        a, b = torch.split(x, self.half, 1)
        return torch.cat((self.IN(a.contiguous()), self.BN(b.contiguous())), 1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, with_ibn=False, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = IBN(planes) if with_ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + (self.downsample(x) if self.downsample is not None else x))


class NonLocal(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.g = nn.Conv2d(c, 1, 1)
        self.theta = nn.Conv2d(c, 1, 1)
        self.phi = nn.Conv2d(c, 1, 1)
        self.W = nn.Sequential(nn.Conv2d(1, c, 1), nn.BatchNorm2d(c))

    def forward(self, x):
        b = x.size(0)
        g = self.g(x).view(b, 1, -1).permute(0, 2, 1)
        theta = self.theta(x).view(b, 1, -1).permute(0, 2, 1)
        phi = self.phi(x).view(b, 1, -1)
        f = torch.matmul(theta, phi)
        y = torch.matmul(f / f.size(-1), g).permute(0, 2, 1).contiguous().view(b, 1, *x.shape[2:])
        return self.W(y) + x


class Backbone(nn.Module):
    LAYERS, NON_LOCAL = [3, 4, 6, 3], [0, 2, 3, 0]   # ResNet-50; non-local after the last 2 / 3 blocks of layer2 / layer3

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.layer1 = self._make(64, 3, 1, True)
        self.layer2 = self._make(128, 4, 2, True)
        self.layer3 = self._make(256, 6, 2, True)
        self.layer4 = self._make(512, 3, 1, False)   # last stride 1: keeps a bigger feature map
        self.NL_2 = nn.ModuleList([NonLocal(512) for _ in range(2)])
        self.NL_3 = nn.ModuleList([NonLocal(1024) for _ in range(3)])
        self.nl_after = {2: [2, 3], 3: [3, 4, 5]}     # block index after which each non-local runs

    def _make(self, planes, blocks, stride, ibn):
        down = None
        if stride != 1 or self.inplanes != planes * 4:
            down = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride, bias=False), nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, ibn, stride, down)]
        self.inplanes = planes * 4
        layers += [Bottleneck(self.inplanes, planes, ibn) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for n, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], start=1):
            k = 0
            for i, block in enumerate(layer):
                x = block(x)
                if i in self.nl_after.get(n, []):
                    x = getattr(self, f"NL_{n}")[k](x)
                    k += 1
        return x


class VehicleReID(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.ones(1, 3, 1, 1))
        self.backbone = Backbone()
        self.gem_p = nn.Parameter(torch.ones(1) * 3)   # GeM power, learned
        self.bottleneck = nn.BatchNorm2d(2048)

    def forward(self, x):
        """x: (N, 3, 256, 256) RGB 0..255 -> (N, 2048)"""
        x = self.backbone((x - self.pixel_mean) / self.pixel_std)
        x = F.adaptive_avg_pool2d(x.clamp(min=1e-6).pow(self.gem_p), 1).pow(1.0 / self.gem_p)
        return self.bottleneck(x)[..., 0, 0]


def load_vehicle_reid(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)["model"]   # weights only: no code is run
    weights = {"pixel_mean": sd["pixel_mean"].view(1, 3, 1, 1), "pixel_std": sd["pixel_std"].view(1, 3, 1, 1),
               "gem_p": sd["heads.pool_layer.p"]}
    weights.update({k: v for k, v in sd.items() if k.startswith("backbone.")})
    weights.update({k.replace("heads.bottleneck.0.", "bottleneck."): v for k, v in sd.items() if k.startswith("heads.bottleneck.0.")})
    net = VehicleReID()
    missing, unexpected = net.load_state_dict(weights, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    assert not missing and not unexpected, (missing, unexpected)
    return net.eval()


if __name__ == "__main__":
    net = load_vehicle_reid("vehicle/models/veri_sbs_R50-ibn.pth")
    with torch.no_grad():
        print(net(torch.rand(2, 3, 256, 256) * 255).shape)
