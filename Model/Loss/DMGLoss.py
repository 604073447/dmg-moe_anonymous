import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.Loss.FocalR import FocalR


# Computes the per-sample focal regression term.
def focal_r_per_sample(pred, target, beta=0.25, gamma=1.0, eps=1e-6):
    err = (pred - target).abs()
    prob = 2 * torch.sigmoid(beta * err) - 1
    return (prob ** gamma) * err


# Combines the main regression loss with DMG-MoE auxiliary losses.
class DMGLoss(nn.Module):
    def __init__(self, loss_kwargs, gate_lambda, aux_lambda, aux_kwargs, eps=1e-8):
        super().__init__()
        self.loss_fn = FocalR(**loss_kwargs)
        self.gate_lambda = gate_lambda
        self.aux_lambda = aux_lambda
        self.aux_kwargs = aux_kwargs
        self.eps = eps

    # Computes the total DMG-MoE training loss.
    def forward(self, outputs, targets, return_dict=False):
        layer_preds = outputs["layer_preds"]
        layer_gates = outputs["layer_gates"]
        y_gt = targets["age"]
        if y_gt.dim() == 1:
            y_gt = y_gt.view(-1, 1)
        main_loss, weight_info = self.loss_fn(outputs, targets)
        total_loss = main_loss
        loss_dict = {
            "main_loss": main_loss.item(),
            "mae": F.l1_loss(outputs["age"], targets["age"]).item(),
        }
        gate_loss = outputs["gate_loss"]
        total_loss = total_loss + gate_loss * self.gate_lambda
        loss_dict["gate_orig"] = gate_loss.item()
        if self.aux_lambda > 0:
            pi = layer_gates.detach()
            pi = pi / pi.sum(dim=1, keepdim=True).clamp_min(self.eps)
            per_sample_losses = focal_r_per_sample(layer_preds, y_gt, **self.aux_kwargs)
            layer_mass = pi.sum(dim=0).clamp_min(self.eps)
            per_expert_loss = (pi * per_sample_losses).sum(dim=0) / layer_mass
            aux_loss = per_expert_loss.mean()
            total_loss = total_loss + self.aux_lambda * aux_loss
            loss_dict["aux_loss"] = aux_loss.item()
        if return_dict:
            loss_dict["loss"] = total_loss.item()
            return total_loss, loss_dict, weight_info
        return total_loss, weight_info
