import torch

from collections import defaultdict
from typing import Dict


@torch.no_grad()
def calculate_metrics_reg(outputs: torch.Tensor, targets: torch.Tensor, eps=1e-8) -> Dict[str, float]:
    outputs = outputs.view(-1)
    targets = targets.view(-1)
    errors = outputs - targets
    abs_errors = errors.abs()

    mae = abs_errors.mean()
    mse = (errors ** 2).mean()
    rmse = torch.sqrt(mse)

    log_abs = torch.log(abs_errors + eps)
    gmean = torch.exp(log_abs.mean())

    metrics = {
        "mae": float(mae.item()),
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "within_3": float((abs_errors <= 3).float().mean().item() * 100),
        "within_5": float((abs_errors <= 5).float().mean().item() * 100),
        "within_10": float((abs_errors <= 10).float().mean().item() * 100),
        "gmean": float(gmean.item()),
    }

    thresholds = torch.arange(1, 11, device=outputs.device)
    cs_mask = abs_errors.unsqueeze(1) <= thresholds.unsqueeze(0)
    metrics["cs"] = float(cs_mask.float().mean(dim=0).mean().item() * 100)
    return metrics


@torch.no_grad()
def calculate_metrics_ord(outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    device = outputs.device
    if outputs.dim() == 2:
        preds = torch.argmax(outputs, dim=1)
    else:
        preds = outputs.long()
    targs = targets.long()

    errors = (preds - targs).float()
    abs_errors = errors.abs()

    mae = abs_errors.mean()
    mse = (errors ** 2).mean()
    rmse = torch.sqrt(mse)

    metrics = {
        "mae": float(mae.item()),
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "accuracy": float((preds == targs).float().mean().item()),
        "within_1": float((abs_errors <= 1).float().mean().item() * 100),
        "within_2": float((abs_errors <= 2).float().mean().item() * 100),
        "within_3": float((abs_errors <= 3).float().mean().item() * 100),
    }

    thresholds = torch.arange(1, 11, device=device)
    cs_mask = abs_errors.unsqueeze(1) <= thresholds.unsqueeze(0)
    metrics["cs"] = float(cs_mask.float().mean(dim=0).mean().item() * 100)
    return metrics


@torch.no_grad()
def calculate_metrics_shot(outputs: torch.Tensor, targets: torch.Tensor,
                           train_labels, many_shot_thr=100, low_shot_thr=20,
                           eps=1e-8):
    device = outputs.device
    train_labels = torch.as_tensor(train_labels, device=device).long()
    targets = targets.long()

    unique_labels = torch.unique(targets)
    num_classes = len(unique_labels)

    train_class_count = torch.zeros(num_classes, device=device, dtype=torch.long)
    test_class_count = torch.zeros(num_classes, device=device, dtype=torch.long)
    mse_per_class = torch.zeros(num_classes, device=device)
    l1_per_class = torch.zeros(num_classes, device=device)

    l1_all_per_class = []
    for idx, l in enumerate(unique_labels):
        train_mask = train_labels == l
        test_mask = targets == l

        train_class_count[idx] = train_mask.sum()
        test_class_count[idx] = test_mask.sum()

        if test_mask.any():
            outputs_l = outputs[test_mask]
            targets_l = targets[test_mask].float()
            errors = outputs_l - targets_l
            abs_errors = errors.abs()

            mse_per_class[idx] = (errors ** 2).sum()
            l1_per_class[idx] = abs_errors.sum()
            l1_all_per_class.append(abs_errors)

    many_mask = train_class_count > many_shot_thr
    low_mask = train_class_count < low_shot_thr
    median_mask = ~(many_mask | low_mask)

    shot = defaultdict(dict)

    def _group(mask, name):
        if mask.any():
            idxs = torch.where(mask)[0]
            abs_errors = torch.cat([l1_all_per_class[i] for i in idxs.tolist()])

            mse = mse_per_class[mask].sum() / test_class_count[mask].sum()
            mae = l1_per_class[mask].sum() / test_class_count[mask].sum()
            rmse = torch.sqrt(mse)
            gmean = torch.exp(torch.log(abs_errors + eps).mean())

            shot[name] = {
                "mae": float(mae.item()),
                "mse": float(mse.item()),
                "rmse": float(rmse.item()),
                "within_3": float((abs_errors <= 3).float().mean().item() * 100),
                "within_5": float((abs_errors <= 5).float().mean().item() * 100),
                "within_10": float((abs_errors <= 10).float().mean().item() * 100),
                "gmean": float(gmean.item()),
            }
        else:
            raise ValueError(f"No samples found for {name} shot group.")

    _group(many_mask, "many")
    _group(median_mask, "median")
    _group(low_mask, "low")
    return shot
