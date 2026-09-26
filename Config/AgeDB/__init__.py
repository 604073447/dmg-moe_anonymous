from dataclasses import dataclass, field

from .data import DataConfig
from .exp import ExperimentConfig
from .model import ModelConfig
from .training import TrainingConfig
from ..directory import PathConfig


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    exp: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    path: PathConfig = field(default_factory=PathConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)



def get_default_config():
    return Config()
