from typing import Dict

import numpy as np
import torch
from torch import autocast
from tqdm import tqdm

from Engine.utils import unpack_batch
from Utils.distributed import is_master, gather_dict, unwrap_model
from Utils.result_output import output_dict
from Utils.variable_duplication import update_dict_to_target, update_dict_to_output


# Initializes per-epoch training statistics.
def init_record(optimizer=None):
    stats = {
        'loss': [],
        'grad': [],
    }
    out_dict = {
        'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
    }
    return stats, out_dict


# Runs the forward pass under autocast and records losses.
def autocast_forward(model, criterion, autocast_kwargs,
                     inputs, targets, stats,
                     epoch=None, idx=None):
    with autocast(**autocast_kwargs):
        outputs = model(inputs, targets['age'], epoch=epoch, idx=idx)
        loss, loss_dict, weight_info = criterion(outputs, targets, return_dict=True)


    update_dict_to_target(stats, loss_dict)
    return outputs, loss, loss_dict, weight_info


def act_grad_norm(model, gradnorm_kwargs, stats, out_dict):
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), **gradnorm_kwargs)
    stats.setdefault('grad', []).append(grad_norm.item())
    out_dict['grad'] = f"{grad_norm.item():.3f}"


# Backpropagates one loss and updates optimizer and EMA state.
def gradnorm_backward(model, optimizer, scaler, grad_scaler_enabled, gradnorm_kwargs, ema_model,
                      loss, stats, out_dict):

    optimizer.zero_grad(set_to_none=True)

    if grad_scaler_enabled:
        if scaler is None:
            raise ValueError('grad_scaler_enabled=True but scaler is None')
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
    else:
        loss.backward()

    if gradnorm_kwargs.get('max_norm', None) is not None:
        act_grad_norm(model, gradnorm_kwargs, stats, out_dict)

    if grad_scaler_enabled:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    if ema_model is not None:
        ema_model.update(unwrap_model(model))


def process_batch_end(out_dict, loss_dict, pbar, show_pbar):
    if show_pbar and is_master():
        out_dict = update_dict_to_output(out_dict, loss_dict, 'f', 3)
        pbar.set_postfix(out_dict)


def process_epoch(stats, desc, logger, epoch):
    result_stats: Dict[str, float] = {}
    for k, v in stats.items():
        if isinstance(v, list) and len(v) > 0:
            result_stats[k] = float(np.mean(v))
        else:

            result_stats[k] = 0.0

    result_stats = gather_dict(result_stats, average=True)
    if is_master() and logger is not None:
        output_dict(result_stats, f"{desc}:  ", logger.info)
    return result_stats


# Trains the model for one epoch.
def train_one_epoch(
    logger, device,
    model, optimizer, criterion, scaler,
    loader, ema_model,
    autocast_kwargs, grad_scaler_enabled, gradnorm_kwargs,
    epoch, global_step,
    desc: str = '',
    show_pbar: bool = True,
):
    model.train()
    desc_str = f"{desc}, Train"
    stats, out_dict = init_record(optimizer)
    pbar = tqdm(loader, desc=desc_str) if (show_pbar and is_master()) else loader
    for idx, batch in enumerate(pbar):
        inputs, targets = unpack_batch(batch, device)
        _, loss, loss_dict, _ = autocast_forward(model, criterion, autocast_kwargs,
                                                 inputs, targets, stats,
                                                 epoch, idx)
        gradnorm_backward(model, optimizer, scaler, grad_scaler_enabled, gradnorm_kwargs, ema_model, loss, stats, out_dict)
        process_batch_end(out_dict, loss_dict, pbar, show_pbar)
        global_step += 1
    result_stats = process_epoch(stats, desc_str, logger, epoch)
    return result_stats, global_step
