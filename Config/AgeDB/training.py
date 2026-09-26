import torch
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    num_epochs: int = 150
    maj_shot: str = "All"
    met_name: str = "MAE"
    find_unused_parameters: bool = False
    ema_on: bool = True
    ema_kwargs: dict = field(default_factory=lambda: {
        "decay": 0.999,
        "use_warmup": True,
        "warmup_step": 10,
    })
    eval_with_ema: bool = True
    loss_kwargs: dict = field(default_factory=lambda: {
        "beta": 0.25,
        "gamma": 2.0,
        "detach_focal_weight": False,
    })
    gate_lambda: float = 0.01
    aux_lambda: float = 0.1
    aux_kwargs: dict = field(default_factory=lambda: {
        "beta": 0.25,
        "gamma": 2.0,
    })
    optimizer_kwargs: dict = field(default_factory=lambda: {
        "lr": 1e-3,
        "weight_decay": 1e-4,
    })
    sec_lr_factor: float = 0.0
    module_treatment: dict = field(default_factory=lambda: {
        "stem": "primary",
        "encoder": "primary",
        "gate": "primary",
        "head": "primary",
    })
    scheduler_kwargs: dict = field(default_factory=lambda: {
        "mode": "min",
        "patience": 2,
        "factor": 0.5,
    })
    autocast_kwargs: dict = field(default_factory=lambda: {
        "device_type": "cuda",
        "enabled": True,
        "dtype": torch.bfloat16,
    })
    grad_scaler_enabled: bool = False
    gradnorm_kwargs: dict = field(default_factory=lambda: {"max_norm": 50.0, "norm_type": 2.0})
    earlystop_kwargs: dict = field(default_factory=lambda: {"patience": 5, "min_delta": 0.0})
    smooth_kwargs: dict = field(default_factory=lambda: {
        "smooth": "EMA",
        "ema_alpha": 0.2,
        "ma_window": 5,
        "use_smoothed_for_best": True,
    })
