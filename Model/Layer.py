import torch.nn as nn

from Model.Encoder import BasicEncoder
from Model.Gate.MAGate import MAGate
from Model.Head import BasicHead


# Combines one encoder stage with its optional gate and expert head.
class Layer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        in_channels: int,
        gate_on: bool,
        head_on: bool,
        is_last_layer: bool,
        encoder_config,
        gate_config,
        layer_bypass_on: bool = True,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.in_channels = in_channels
        self.out_channels = None
        self.gate_on = gate_on
        self.head_on = head_on
        self.is_last_layer = is_last_layer
        self.encoder_config = encoder_config
        self.gate_config = gate_config
        self.encoder = None
        self.pool = None
        self.gate = None
        self.head = None
        self.layer_bypass_on = layer_bypass_on
        self._initialize()

    def _initialize(self):
        self.encoder = BasicEncoder(
            eid=self.layer_id,
            in_channels=self.in_channels,
            is_last_stage=self.is_last_layer,
            layer_bypass_on=self.layer_bypass_on,
            **vars(self.encoder_config),
        )
        self.out_channels = self.encoder.out_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = MAGate(
            gid=self.layer_id,
            in_channels=self.encoder.out_channels,
            **self.gate_config.kwargs,
        ) if self.gate_on else None
        self.head = BasicHead(
            hid=self.layer_id,
            in_channels=self.encoder.out_channels,
        ) if self.head_on else None

    # Returns stage embedding, pooled feature, expert output, and gate output.
    def forward(self, x, y_gt=None, compute_gate=True, epoch=None, idx=None):
        emb = self.encoder(x)
        z = self.pool(emb).view(emb.size(0), -1)

        head_out = None
        if self.head_on and self.head is not None:
            head_out = self.head(z)

        gate_out = None
        if compute_gate and self.gate_on and self.gate is not None:
            gate_out = self.gate(z, y_gt=y_gt, epoch=epoch, idx=idx)

        return emb, z, head_out, gate_out
