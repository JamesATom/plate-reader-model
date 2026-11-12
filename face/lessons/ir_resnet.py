"""IR-ResNet: the network behind AdaFace (and ArcFace). Plain PyTorch, no training code.

Trimmed copy of models/iresnet/model.py from minchul/cvlface_adaface_ir101_webface12m
(AdaFace by Minchul Kim, MIT License). Only the parts needed to run IR-18/50/101.
"""

import torch
from torch import nn


class BasicBlockIR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                                                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth))

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


UNITS = {18: [2, 2, 2, 2], 50: [3, 4, 14, 3], 100: [3, 13, 30, 3]}


class IResNet(nn.Module):
    """112x112 RGB face (normalised to -1..1) -> 512 numbers."""

    def __init__(self, num_layers=100, output_dim=512):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64))
        modules, in_c = [], 64
        for depth, n in zip([64, 128, 256, 512], UNITS[num_layers]):
            modules.append(BasicBlockIR(in_c, depth, 2))
            modules += [BasicBlockIR(depth, depth, 1) for _ in range(n - 1)]
            in_c = depth
        self.body = nn.Sequential(*modules)
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout(0.4), nn.Flatten(),
                                          nn.Linear(512 * 7 * 7, output_dim),
                                          nn.BatchNorm1d(output_dim, affine=False))

    def forward(self, x):
        return self.output_layer(self.body(self.input_layer(x)))


def load_adaface(path, num_layers=100):
    from safetensors.torch import load_file
    weights = {k.removeprefix("model.net."): v for k, v in load_file(path).items()}
    net = IResNet(num_layers)
    net.load_state_dict(weights)
    return net.eval()


if __name__ == "__main__":
    net = load_adaface("face/models/adaface_ir101/model.safetensors")
    print(net(torch.zeros(2, 3, 112, 112)).shape)
