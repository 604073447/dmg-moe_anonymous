from dataclasses import dataclass, field

from .Module.encoder import EncoderConfig
from .Module.gate import GateConfig
from .Module.stem import StemConfig


@dataclass
class LayerConfig:
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    gate_config: GateConfig = field(default_factory=GateConfig)


@dataclass
class ModelConfig:
    in_channels: int = 3
    stem_channels: int = 64
    stem_config: StemConfig = field(default_factory=StemConfig)
    layer_config: LayerConfig = field(default_factory=LayerConfig)
    structure_define: dict = field(default_factory=lambda: {
        "num_layers": 4,
        "num_encoder_blocks_per_layer": [3, 4, 6, 3],
    })
    layer_bypass_on: bool = True
