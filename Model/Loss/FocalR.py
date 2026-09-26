import torch
import torch.nn as nn


class FocalR(nn.Module):
    def __init__(self, beta: float = 1.0, gamma: float = 2.0,
                 reduction='mean',
                 eps: float = 1e-8,
                 detach_focal_weight: bool = False,
                 normalize_by: str = 'weights'):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.eps = float(eps)
        self.detach_focal_weight = bool(detach_focal_weight)

        allowed = {'weights', 'focal_and_weights', 'batch'}
        if normalize_by not in allowed:
            raise ValueError(f"normalize_by must be one of {sorted(allowed)}, got: {normalize_by}")
        self.normalize_by = normalize_by

    def forward(self, outputs, targets):
        output = outputs['age']
        target = targets['age']
        if output.shape[0] != target.shape[0]:
            raise ValueError(f"Batch size mismatch: outputs['age'] has {output.shape[0]} samples, "
                             f"targets['age'] has {target.shape[0]} samples")
        B = int(output.shape[0])
        element_wise_loss = torch.abs(output - target)
        abs_error = torch.abs(output - target)
        modulating_factor = (2 * torch.sigmoid(self.beta * abs_error) - 1) ** self.gamma
        
        if self.detach_focal_weight:
            modulating_factor = modulating_factor.detach()
        loss = element_wise_loss * modulating_factor

        sample_weight = targets.get('weight', None)
        if sample_weight is None:
            sample_weight = torch.ones((B,), device=output.device, dtype=loss.dtype)
        else:
            sample_weight = sample_weight.to(device=output.device, dtype=loss.dtype).view(-1)
            if sample_weight.numel() != B:
                raise ValueError(f"targets['weight'] must have {B} elements, got {sample_weight.numel()}")

        path_weight = outputs.get('target_weights', None)
        if path_weight is None:
            path_weight = torch.ones_like(sample_weight)
        else:
            path_weight = path_weight.to(device=output.device, dtype=loss.dtype).view(-1)
            if path_weight.numel() != B:
                raise ValueError(f"outputs['target_weights'] must have {B} elements, got {path_weight.numel()}")

        weights = sample_weight * path_weight
        if loss.dim() > 1:
            weights_view = weights.view(-1, 1)
        else:
            weights_view = weights

        loss = loss * weights_view
        weight_info = {
            'sample_weight': sample_weight,
            'path_weight': path_weight,
            'focal_weight': modulating_factor,
        }

        if self.reduction == 'none':
            return loss, weight_info
        elif self.reduction == 'sum':
            return torch.sum(loss, dim=0), weight_info
        elif self.reduction == 'mean':
            num = torch.sum(loss)
            if self.normalize_by == 'weights':
                den = weights.sum().clamp_min(self.eps)
            elif self.normalize_by == 'focal_and_weights':
                fw = modulating_factor
                if fw.dim() > 1:
                    fw = fw.view(fw.shape[0], -1).mean(dim=1)
                den = (weights * fw).sum().clamp_min(self.eps)
            elif self.normalize_by == 'batch':
                den = max(int(output.shape[0]), 1)
            else:
                raise RuntimeError(f"Unexpected normalize_by: {self.normalize_by}")
            return num / den, weight_info
        else:
            raise NotImplementedError(f'Unsupported reduction mode: {self.reduction}')
