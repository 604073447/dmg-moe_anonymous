from dataclasses import dataclass, field


@dataclass
class StemConfig:
    kwargs: dict = field(default_factory=lambda: {
        "norm_type": "BatchNorm2d",
        "act_name": "ReLU",
    })
