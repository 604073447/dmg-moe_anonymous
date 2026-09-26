import copy

import torch.nn as nn

from Model.Block.Bottleneck import Bottleneck
from Model.utils import norm_resolver


# Builds one ResNet encoder stage from bottleneck blocks.
class BasicEncoder(nn.Module):
    def __init__(self, eid, in_channels, stride, channel_factor, num_blocks, block_config, is_last_stage=False, layer_bypass_on=True):
        super().__init__()
        self.eid = eid
        self.in_channels = in_channels
        self.stride = stride
        self.channel_factor = channel_factor ** self.eid
        self.num_blocks = num_blocks
        self.block_config = block_config
        self.is_last_stage = is_last_stage
        self.layer_bypass_on = layer_bypass_on
        self.block_kwargs = None
        self.base_channels = None
        self.out_channels = None
        self.block_list = nn.ModuleList()
        self._initialize()

    def _initialize(self):
        self.block_kwargs = copy.deepcopy(self.block_config.kwargs)
        if "base_channels" not in self.block_kwargs:
            raise ValueError("Bottleneck blocks must define base_channels.")
        scaled_base = int(self.block_kwargs["base_channels"] * self.channel_factor)
        self.block_kwargs["base_channels"] = scaled_base
        self.base_channels = scaled_base
        current_channels = self.in_channels
        for _ in range(self.num_blocks):
            block = self._create_block(current_channels)
            self.block_list.append(block)
            current_channels = block.out_channels
        self.out_channels = current_channels

    def _create_block(self, in_channels):
        is_first = len(self.block_list) == 0
        block_kwargs = copy.deepcopy(self.block_kwargs)
        stride = self.stride if is_first else 1
        out_channels = int(block_kwargs["base_channels"] * block_kwargs["expansion"])
        if self.layer_bypass_on and (stride != 1 or in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                norm_resolver(self.block_config.downsample_norm, out_channels),
            )
        else:
            downsample = None
        block = Bottleneck(
            in_channels,
            stride,
            downsample,
            bypass_on=self.layer_bypass_on or not is_first,
            **block_kwargs,
        )
        return block

    # Applies the stage blocks in order.
    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x
