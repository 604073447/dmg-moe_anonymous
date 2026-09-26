import torch.nn as nn

from Model.utils import activation_resolver, norm_resolver


# Builds the final ResNet input stem.
class ResNetStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64,
                 norm_type='BatchNorm2d', act_name='ReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels//2,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_resolver(norm_type, out_channels//2)
        self.act1 = activation_resolver(act_name)

        self.conv2 = nn.Conv2d(out_channels//2, out_channels//2,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_resolver(norm_type, out_channels//2)
        self.act2 = activation_resolver(act_name)

        self.conv3 = nn.Conv2d(out_channels//2, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = norm_resolver(norm_type, out_channels)
        self.act3 = activation_resolver(act_name)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.maxpool(x)
        return x
