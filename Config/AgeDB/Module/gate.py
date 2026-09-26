from dataclasses import dataclass, field


@dataclass
class GateConfig:
    kwargs: dict = field(default_factory=lambda: {
        "proj_cfg": {
            "hidden_dim": 64,
            "num_layers": 1,
            "mlp_kwargs": {
                "norm_type": "BatchNorm1d",
                "act_name": "ReLU",
                "dropout": 0.0,
            },
        },
        "tau_y": 10.0,
        "tau_z": 0.08,
        "detach_gate_input": True,
        "detach_z_for_gloss": True,
        "sample_unique_age": True,
        "norm_z_logits": True,
    })
