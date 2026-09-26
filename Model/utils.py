import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def activation_resolver(name: str):
    if name is None or name == 'None':
        return None
    elif name == 'ReLU':
        return nn.ReLU(inplace=True)
    elif name == 'ReLU6':
        return nn.ReLU6(inplace=True)
    elif name == 'SiLU':
        return nn.SiLU(inplace=True)
    elif name == 'LeakyReLU':
        return nn.LeakyReLU(inplace=True)
    elif name == 'GELU':
        return nn.GELU()
    elif name == 'ELU':
        return nn.ELU(inplace=True)
    elif name == 'Sigmoid':
        return nn.Sigmoid()
    elif name == 'Tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation type: {name}")


def norm_resolver(name: str, num_channels: int, gn_groups: int = 32):
    if name is None or name == 'None':
        return None
    elif name == 'BatchNorm1d':
        return nn.BatchNorm1d(num_channels)
    elif name == 'LayerNorm1d':
        return nn.LayerNorm(num_channels)
    elif name == 'BatchNorm2d':
        return nn.BatchNorm2d(num_channels)
    elif name == 'GroupNorm':
        g = min(gn_groups, num_channels)
        if num_channels % g != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by gn_groups ({g}) for GroupNorm.")
        return nn.GroupNorm(g, num_channels)
    elif name == 'LayerNorm2d':
        return nn.GroupNorm(1, num_channels)
    else:
        raise ValueError(f"Unsupported normalization type: {name}")
