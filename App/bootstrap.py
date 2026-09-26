import copy
from collections import deque
import math
import re

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from Dataset.DALIDataset import get_dali_dataloader
from Dataset.utils import resolve_binning_params
from Model.ResNetDMGMoE import ResNetDMGMoE
from Utils.distributed import is_master, replace_sync_bn, log_info
from Utils.file_access import save_config, load_config
from Utils.predefine import SPACE4, PARAM_PREFIX
from Utils.variable_extract import get_unique_top_k_levels


# Loads configuration and applies command-line overrides.
def build_config(logger, args, exp_dir, timestamp):
    config = load_config()
    config.path.exp_dir = exp_dir
    eval_bins = resolve_binning_params(**config.data.eval_bins)
    config.data.eval_bins.update(eval_bins)
    bins = np.arange(0, eval_bins["max_age"], eval_bins["bin_step"])[:-1]
    config.data.label_map_dict["bins"] = bins
    config.data.label_map_dict["label_bin_map"] = np.digitize([i for i in range(eval_bins["max_age"])], bins) - 1
    config.data.label_map_dict["bin_name_map"] = np.array([f"[{l}, {r})" for l, r in zip(bins[:-1], bins[1:])] + [f"[{bins[-1]}, {eval_bins['max_age']-1}]"])
    train_label = pd.read_csv(config.data.train_csv)[config.data.label_col].to_numpy(dtype=int)
    counts = np.bincount(train_label, minlength=config.data.max_age)
    label_shot_map = np.full_like(np.arange(config.data.max_age), 2, dtype=int)
    label_shot_map[counts >= config.data.shot_thr_dict["many_shot_thr"]] = 3
    label_shot_map[counts <= config.data.shot_thr_dict["low_shot_thr"]] = 1
    config.data.label_map_dict["shot_thr_dict"] = config.data.shot_thr_dict
    config.data.label_map_dict["label_shot_map"] = label_shot_map
    config.data.label_map_dict["shot_name_map"] = np.array(["Low Shot", "Med. Shot", "Many Shot"])
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.lr_scale is not None:
        log_info(logger, f"{SPACE4}Scaling learning rates by '{args.lr_scale}' due to distributed training...")
        if args.lr_scale == "none":
            scale_factor = 1.0
        elif args.lr_scale == "linear":
            scale_factor = dist.get_world_size() if dist.is_initialized() else 1.0
        elif args.lr_scale == "sqrt":
            scale_factor = math.sqrt(dist.get_world_size()) if dist.is_initialized() else 1.0
        else:
            raise ValueError(f"Unknown lr_scale: {args.lr_scale}")
        update_scaled_lr(config.train, scale_factor)
    else:
        log_info(logger, f"{SPACE4}No learning rate scaling applied.")
    if args.seed is not None:
        config.exp.seed = args.seed
    if is_master():
        save_config(config, exp_dir, timestamp)
    return config


def update_scaled_lr(cfg, scale_factor):
    cfg.optimizer_kwargs["lr"] *= scale_factor


# Builds the DALI loader for one data split.
def build_loader(mode, config, logger, ddp_info):
    cfg = config.data
    is_training = mode == "train"
    drop_last = bool(is_training and cfg.drop_last)
    loader = get_dali_dataloader(
        mode=mode,
        logger=logger,
        csv_file=getattr(cfg, f"{mode}_csv"),
        img_dir=cfg.img_dir,
        crop_size=cfg.img_size,
        is_training=is_training,
        batch_size=cfg.batch_size,
        num_threads=cfg.num_workers,
        drop_last=drop_last,
        dali_setup=cfg.dali_setup,
        augment_kwargs=cfg.augment,
        max_age=cfg.max_age,
        reweight_kwargs=cfg.reweight,
        lds_kwargs=cfg.lds,
        shot_thr_dict=cfg.shot_thr_dict,
        ddp_info=ddp_info,
        base_seed=config.exp.seed or 42,
        label_col=cfg.label_col,
        path_col=cfg.path_col,
    )
    return None, loader


# Instantiates the ResNet DMG-MoE model.
def build_model(config, device, ddp_info):
    model = ResNetDMGMoE(**vars(config.model))
    if is_master():
        print(f"Model built:\n{model}")
    if ddp_info["world_size"] > 1 and config.exp.use_sync_bn:
        model = replace_sync_bn(model)
    model = model.to(device)
    if config.exp.torch_compile:
        model = model.to(memory_format=torch.channels_last)
        model = torch.compile(model, mode="max-autotune")
    return model


# Creates the training loss used by DMG-MoE.
def build_criterion(cfg):
    from Model.Loss.DMGLoss import DMGLoss
    return DMGLoss(
        cfg.loss_kwargs,
        cfg.gate_lambda,
        cfg.aux_lambda,
        cfg.aux_kwargs,
    )


# Creates optimizer parameter groups from module treatments.
def build_optimizer(cfg, phase, model, logger=None):
    kwargs = copy.deepcopy(cfg.optimizer_kwargs)
    base_lr = kwargs.pop("lr")
    primary_pattern, secondary_pattern, freeze_pattern = group_pattern(cfg, phase)
    sec_lr_factor = getattr(cfg, "sec_lr_factor", 0.0)
    freeze_sec = sec_lr_factor == 0.0
    pri_names, pri_params, sec_names, sec_params, fre_names, fre_params = get_param_list(model, primary_pattern, secondary_pattern, freeze_pattern, freeze_sec)
    if len(pri_names) == 0:
        raise ValueError(f"No primary parameters matched for optimizer in phase '{phase}'.")
    param_groups = [{"params": pri_params, "lr": base_lr}]
    if not freeze_sec and len(sec_names) > 0:
        param_groups.append({"params": sec_params, "lr": base_lr * sec_lr_factor})
    if logger is not None and is_master():
        output_str = f"\nOptimizer parameter groups for phase '{phase}':\n"
        output_str += f"  Pri params (lr={base_lr}):\n"
        for name in get_unique_param_names(pri_names):
            output_str += f"    {name}\n"
        if not freeze_sec and len(sec_names) > 0:
            output_str += f"  Sec params (lr={base_lr * sec_lr_factor}):\n"
            for name in get_unique_param_names(sec_names):
                output_str += f"    {name}\n"
        if len(fre_names) > 0:
            output_str += "  Special freeze params:\n"
            for name in get_unique_param_names(fre_names):
                output_str += f"    {name}\n"
        logger.info(output_str)
    return torch.optim.AdamW(param_groups, **kwargs)


def group_pattern(cfg, phase):
    if phase != "Init":
        raise ValueError(f"Unknown phase for optimizer: {phase}")
    pattern_list = {
        "stem": [re.compile(PARAM_PREFIX + r"stem\.")],
        "encoder": [re.compile(PARAM_PREFIX + r"layers\.(\d+)\.encoder\.")],
        "gate": [re.compile(PARAM_PREFIX + r"layers\.(\d+)\.gate\.")],
        "head": [re.compile(PARAM_PREFIX + r"layers\.(\d+)\.head\.")],
    }
    primary_pattern = []
    secondary_pattern = []
    freeze_pattern = []
    for key, patterns in pattern_list.items():
        treatment = cfg.module_treatment[key]
        if treatment == "primary":
            primary_pattern.extend(patterns)
        elif treatment == "secondary":
            secondary_pattern.extend(patterns)
        elif treatment == "freeze":
            freeze_pattern.extend(patterns)
        else:
            raise ValueError(f"Unknown treatment '{treatment}' for module '{key}'.")
    return primary_pattern, secondary_pattern, freeze_pattern


def get_param_list(model, pri_pattern, sec_pattern, fre_pattern, freeze_sec=False):
    pri_compiled = compile_pattern(pri_pattern)
    sec_compiled = compile_pattern(sec_pattern)
    fre_compiled = compile_pattern(fre_pattern)
    pri_names, pri_params, sec_names, sec_params, fre_names, fre_params = [], [], [], [], [], []
    for name, param in model.named_parameters():
        pri_is_match = any(p.search(name) for p in pri_compiled)
        sec_is_match = any(p.search(name) for p in sec_compiled)
        fre_is_match = any(p.search(name) for p in fre_compiled)
        if pri_is_match:
            param.requires_grad = True
            pri_names.append(name)
            pri_params.append(param)
        elif sec_is_match:
            param.requires_grad = not freeze_sec
            sec_names.append(name)
            sec_params.append(param)
        elif fre_is_match:
            param.requires_grad = False
            fre_names.append(name)
            fre_params.append(param)
        else:
            param.requires_grad = False
    return pri_names, pri_params, sec_names, sec_params, fre_names, fre_params


def compile_pattern(pattern):
    return [re.compile(p) if isinstance(p, str) else p for p in pattern]


# Summarizes parameter names for optimizer logging.
def get_unique_param_names(paths):
    unique_prefixes = set()
    for path in paths:
        parts = path.split(".")
        k = 3 if parts[0] == "layers" else 2
        unique_prefixes.add(".".join(parts[:k]))
    return sorted(unique_prefixes)


# Creates the learning-rate scheduler.
def build_scheduler(cfg, optimizer, num_epochs=None):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler_kwargs)


# Creates the mixed-precision gradient scaler.
def build_normscale(cfg):
    return torch.amp.GradScaler(device=cfg.autocast_kwargs["device_type"], enabled=cfg.grad_scaler_enabled)


def build_earlystop(cfg, best_record):
    return EarlyStopping(best_epoch=best_record["epoch"], best_score=best_record["score"], mode=best_record["mode"], **cfg.earlystop_kwargs, **cfg.smooth_kwargs)


class EarlyStopping:
    def __init__(self, stage="", patience=5, min_delta=0.0, mode="min", verbose=False, best_epoch=None, best_score=None, smooth=None, ema_alpha=0.2, ma_window=5, use_smoothed_for_best=True):
        self.stage = stage
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        if smooth not in (None, "None", "EMA", "MA"):
            raise ValueError(f"smooth must be None/'EMA'/'MA', got {smooth}")
        self.smooth = smooth
        self.ema_alpha = float(ema_alpha)
        self.ma_window = int(ma_window)
        self.use_smoothed_for_best = bool(use_smoothed_for_best)
        self.best_epoch = best_epoch
        self.best_score = best_score
        self.best_raw_score = None
        self.best_test_score = None
        self.counter = 0
        self.early_stop = False
        self.improved = False
        self.history = []
        self.smooth_history = []
        self._ema_state = None
        self._ma_buf = deque(maxlen=self.ma_window)

    def _is_improvement(self, current_score):
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current_score < (self.best_score - self.min_delta)
        return current_score > (self.best_score + self.min_delta)

    def _compute_smoothed(self, raw_score):
        if self.smooth is None or self.smooth == "None":
            return float(raw_score)
        if self.smooth == "EMA":
            if self._ema_state is None:
                self._ema_state = float(raw_score)
            else:
                self._ema_state = self.ema_alpha * float(raw_score) + (1.0 - self.ema_alpha) * self._ema_state
            return float(self._ema_state)
        self._ma_buf.append(float(raw_score))
        return float(np.mean(self._ma_buf))

    def __call__(self, current_score, current_epoch=None, test_score=None):
        if not np.isfinite(current_score):
            raise ValueError(f"Non-finite score detected: {current_score}")
        raw = float(current_score)
        smoothed = self._compute_smoothed(raw)
        self.history.append(raw)
        self.smooth_history.append(smoothed)
        self.improved = False
        monitored = smoothed if self.use_smoothed_for_best else raw
        if self._is_improvement(monitored):
            self.best_score = float(monitored)
            self.best_epoch = current_epoch
            self.best_raw_score = raw
            self.best_test_score = float(test_score) if test_score is not None else None
            self.counter = 0
            self.improved = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def build_ema_model(cfg, model, device):
    return ModelEMA(model, device=device, **cfg.ema_kwargs) if cfg.ema_on else None


class ModelEMA:
    def __init__(self, model, decay=0.999, use_warmup=True, warmup_step=10, device=None):
        self.decay = float(decay)
        self.use_warmup = use_warmup
        self.warmup_step = warmup_step
        self.updates = 0
        self.ema = copy.deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)
        if device is not None:
            self.ema.to(device=device)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        if self.use_warmup:
            decay = min(self.decay, (1.0 + self.updates) / (self.warmup_step + self.updates))
        else:
            decay = self.decay
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()
        for key, value in ema_state.items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(model_state[key].detach().to(value.dtype), alpha=(1.0 - decay))
            else:
                ema_state[key].copy_(model_state[key])

    def state_dict(self):
        return {
            "decay": self.decay,
            "use_warmup": self.use_warmup,
            "warmup_step": self.warmup_step,
            "updates": self.updates,
            "ema": self.ema.state_dict(),
        }

    def load_state_dict(self, state):
        self.decay = state.get("decay", self.decay)
        self.use_warmup = state.get("use_warmup", self.use_warmup)
        self.warmup_step = state.get("warmup_step", self.warmup_step)
        self.updates = state.get("updates", 0)
        self.ema.load_state_dict(state["ema"], strict=True)
