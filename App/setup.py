import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from Utils.distributed import setup_distributed, log_info, is_master
from Utils.predefine import SPACE4


def setup_experiment(args):
    timestamp = pick_timestamp()
    exp_dir = os.environ.get("EXPERIMENT_DIR", args.exp_dir or f"Results/{timestamp}")
    if is_master():
        Path(exp_dir).mkdir(parents=True, exist_ok=True)
    ddp_info = setup_distributed() if args.distributed else {"rank": 0, "world_size": 1, "local_rank": 0}
    if is_master():
        Path(exp_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger("AgeDB", f"{exp_dir}/AgeDB.log") if is_master() else None
    log_info(logger, f"Training started at {timestamp}")
    log_info(logger, "Building Experiment...")
    device = torch.device(f"cuda:{ddp_info['local_rank']}" if torch.cuda.is_available() else "cpu")
    log_info(logger, f"{SPACE4}Visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}, current device: {device}")
    if args.seed is not None:
        seed_everything(args.seed)
        log_info(logger, f"{SPACE4}Random seed set to: {args.seed}")
    else:
        log_info(logger, f"{SPACE4}Random seed not set")
    return timestamp, exp_dir, ddp_info, logger, device


def pick_timestamp():
    if "EXPERIMENT_TIMESTAMP" in os.environ:
        return os.environ["EXPERIMENT_TIMESTAMP"]
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def get_logger(name, filepath=None, level=logging.INFO, mode="w"):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s -  %(message)s")
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, mode)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
