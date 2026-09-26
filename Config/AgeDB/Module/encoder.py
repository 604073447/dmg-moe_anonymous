from dataclasses import dataclass, field


@dataclass
class EncoderBlockConfig:
    kwargs: dict = field(default_factory=lambda: {
        "base_channels": 64,
        "expansion": 4,
        "norm_type": "BatchNorm2d",
        "act_name": "ReLU",
    })
    downsample_norm: str = "BatchNorm2d"


@dataclass
class EncoderConfig:
    stride: int = 2
    channel_factor: float = 2.0
    num_blocks: int = None
    block_config: EncoderBlockConfig = field(default_factory=EncoderBlockConfig)
