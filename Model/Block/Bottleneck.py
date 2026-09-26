import torch.nn as nn

from Model.utils import activation_resolver, norm_resolver


# Implements the ResNet bottleneck block used by DMG-MoE.
class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: int = 1,
        downsample=None,
        base_channels: int = 64,
        expansion: int = 4,
        norm_type: str = 'BatchNorm2d',
        act_name: str = 'ReLU',
        bypass_on: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.base_channels = base_channels
        self.out_channels = base_channels * expansion
        self.expansion = expansion
        self.bypass_on = bypass_on
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, base_channels, 1, bias=False)
        self.bn1 = norm_resolver(norm_type, base_channels)
        self.act1 = activation_resolver(act_name)

        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, stride, 1, bias=False)
        self.bn2 = norm_resolver(norm_type, base_channels)
        self.act2 = activation_resolver(act_name)

        self.conv3 = nn.Conv2d(base_channels, self.out_channels, 1, bias=False)
        self.bn3 = norm_resolver(norm_type, self.out_channels)
        self.act3 = activation_resolver(act_name)

    def forward(self, x):
        if self.bypass_on:
            identity = self.downsample(x) if self.downsample is not None else x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.bypass_on:
            out += identity
        return self.act3(out)
