from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    seed: Optional[int] = 42
    auto_resume: bool = False
    use_sync_bn: bool = True
    torch_compile: bool = False
