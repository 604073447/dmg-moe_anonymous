from typing import Dict, Optional, List
import torch
import torch.nn as nn
import copy

from Model.Layer import Layer
from Model.Stem.ResNetStem import ResNetStem
from Model.utils import init_weights


# Implements the ResNet DMG-MoE backbone and expert fusion.
class ResNetDMGMoE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        stem_config,
        layer_config,
        structure_define: dict,
        layer_bypass_on: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.stem_config = copy.deepcopy(stem_config)
        self.layer_config = copy.deepcopy(layer_config)
        self.structure_define = copy.deepcopy(structure_define)
        self.layer_bypass_on = layer_bypass_on

        self.stem = None
        self.layers = nn.ModuleList()
        self._initialize()

    # Builds model modules and initializes parameters.
    def _initialize(self):
        self.validate_structure_define()
        self.build_stem()
        self.build_fixed_architecture()
        self._initialize_weights()

    # Checks that the configured stage layout is valid.
    def validate_structure_define(self):
        num_layers = self.structure_define['num_layers']
        block_layout = self.structure_define['num_encoder_blocks_per_layer']
        if len(block_layout) != num_layers:
            raise ValueError(
                f"Expected {num_layers} entries in 'num_encoder_blocks_per_layer', got {len(block_layout)}."
            )

    # Builds the ResNet input stem.
    def build_stem(self):
        self.stem = ResNetStem(
            in_channels=self.in_channels,
            out_channels=self.stem_channels,
            **self.stem_config.kwargs,
        )

    # Builds the fixed multi-depth expert architecture.
    def build_fixed_architecture(self):
        num_layers = self.structure_define['num_layers']
        in_channels = self.stem_channels
        for layer_idx in range(num_layers):
            layer_config = copy.deepcopy(self.layer_config)
            layer_config.encoder_config.num_blocks = self.structure_define['num_encoder_blocks_per_layer'][layer_idx]
            layer = Layer(
                layer_id=layer_idx,
                in_channels=in_channels,
                gate_on=layer_idx < num_layers - 1,
                head_on=True,
                is_last_layer=(layer_idx == num_layers - 1),
                layer_bypass_on=self.layer_bypass_on,
                **vars(layer_config),
            )
            self.layers.append(layer)
            in_channels = layer.out_channels

    # Applies the configured module initialization.
    def _initialize_weights(self) -> None:
        for m in self.modules():
            init_weights(m)

    # Computes layer predictions, gates, and final DMG-MoE output.
    def forward(self, x,
                y_gt = None,
                return_emb: bool = False,
                return_layer_feats: bool = False,
                epoch=None, idx=None,):

        emb = self.stem(x) if self.stem is not None else x

        final_out: Optional[Dict] = {}
        final_emb: Optional[Dict] = {}

        layer_feats_list: List[torch.Tensor] = []
        layer_preds_list: List[torch.Tensor] = []
        layer_gates_list: List[torch.Tensor] = []
        layer_gloss_list: List[torch.Tensor] = []

        for l_i, layer in enumerate(self.layers):
            emb, z, head_out, _ = layer(emb, y_gt=y_gt, compute_gate=False, epoch=epoch, idx=idx)
            layer_feats_list.append(z)

            if head_out is not None and 'age' in head_out:
                layer_preds_list.append(head_out['age'])

            if l_i == len(self.layers) - 1:
                final_out = head_out
                final_emb = z
                final_out['emb'] = final_emb

        for l in range(len(self.layers) - 1):
            gate = self.layers[l].gate
            if gate is None:
                raise RuntimeError(f'Expect gate at layer {l} for DMG-MoE stick_breaking mode.')
            gate_out = gate(layer_feats_list[l], y_gt=y_gt, epoch=epoch, idx=idx)

            layer_gates_list.append(gate_out['gate_prob'])
            layer_gloss_list.append(gate_out.get('gate_loss', torch.tensor(0.0, device=emb.device)))

        layer_preds = torch.stack(layer_preds_list, dim=1)
        gate_gloss = torch.stack(layer_gloss_list).mean()

        if len(layer_preds_list) != len(self.layers):
            raise RuntimeError(
                f'DMG-MoE expects a head at each depth. '
                f'Got {len(layer_preds_list)} expert preds for {len(self.layers)} layers.'
            )
        if len(layer_gates_list) != len(self.layers) - 1:
            raise RuntimeError(
                f'Expect {len(self.layers) - 1} gates. '
                f'Got {len(layer_gates_list)}.'
            )

        layer_gates = self._stick_breaking(layer_gates_list)
        layer_gates = layer_gates / layer_gates.sum(dim=1, keepdim=True)
        final_pred = (layer_gates * layer_preds).sum(dim=1)

        final_out = {
            'age': final_pred,
            'emb': final_emb,
            'gate_loss': gate_gloss,
            'layer_preds': layer_preds,
            'layer_gates': layer_gates,
            'layer_feats': layer_feats_list,
        }

        if return_emb and return_layer_feats:
            return final_out, final_emb, layer_feats_list
        if return_emb:
            return final_out, final_emb
        if return_layer_feats:
            return final_out, layer_feats_list
        return final_out

    # Converts gate probabilities to stick-breaking expert weights.
    @staticmethod
    def _stick_breaking(layer_gates: List[torch.Tensor], eps=1e-12) -> torch.Tensor:
        if len(layer_gates) == 0:
            raise ValueError('exit_probs must be non-empty')

        remain = torch.ones((layer_gates[0].shape[0],),
                            device=layer_gates[0].device,
                            dtype=layer_gates[0].dtype)

        weights = []
        for p in layer_gates:
            p = p.clamp(eps, 1.0 - eps)
            w = remain * p
            weights.append(w)
            remain = (remain - w).clamp_min(0.0)
        weights.append(remain)
        return torch.stack(weights, dim=1)
