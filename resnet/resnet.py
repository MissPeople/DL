# Author    : zhiping
import torch
import torch.nn as nn
from torch.nn import functional as F, Flatten, Linear
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d


class ResidualModule(nn.Module):
    """
        this module is common module, like MLP
    """
    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(ResidualModule, self).__init__()
        self.common = Sequential(
            Conv2d(in_channel, out_channel, kernel, stride, padding),
            BatchNorm2d(out_channel),
            ReLU(),
            Conv2d(out_channel, out_channel, kernel, stride, padding),
            BatchNorm2d(out_channel)
        )

    def forward(self, x):
        identity = x
        x = self.common(x)
        out = F.relu(identity + x)
        return out


class SpecialModule(nn.Module):
    """
        This module mainly uses 1*1 convolutional layers to achieve upscaling and downscaling.
    """
    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(SpecialModule, self).__init__()
        self.normal = Sequential(
            Conv2d(in_channel, out_channel, kernel[0], stride[0], padding),
            BatchNorm2d(out_channel),
            ReLU(),
            Conv2d(out_channel, out_channel, kernel[0], stride[1], padding),
            BatchNorm2d(out_channel)
        )
        self.downsample = Sequential(
            Conv2d(in_channel, out_channel, kernel[1], stride[0]),
            BatchNorm2d(out_channel)
        )

    def forward(self, x):
        _x = self.downsample(x)
        x = self.normal(x)
        x = F.relu(_x + x)
        return x


class ResNet18(nn.Module):
    """
        this module apply size is 224*224
    """
    def __init__(self):
        super(ResNet18, self).__init__()
        self.prepare = Sequential(
            Conv2d(3, 64, 7, 2, 3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(3, 2, 1)
        )
        self.layer1 = Sequential(
            ResidualModule(64, 64, 3, 1, 1),
            ResidualModule(64, 64, 3, 1, 1)
        )
        self.layer2 = Sequential(
            SpecialModule(64, 128, [3, 1], [2, 1], 1),
            ResidualModule(128, 128, 3, 1, 1)
        )
        self.layer3 = Sequential(
            SpecialModule(128, 256, [3, 1], [2, 1], 1),
            ResidualModule(256, 256, 3, 1, 1)
        )
        self.layer4 = Sequential(
            SpecialModule(256, 512, [3, 1], [2, 1], 1),
            ResidualModule(512, 512, 3, 1, 1)
        )
        self.out = Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            Linear(512, 10)
        )

    def forward(self, x):
        x = self.prepare(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out(x)
        return x


if __name__ == "__main__":
    net = ResNet18()
    x = torch.randn(2, 3, 224, 224)
    print(x.shape)
    y = net(x)
    print(y.shape)
