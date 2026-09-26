import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import torch
from Config.AgeDB import get_default_config


def make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_config():
    return get_default_config()


def _to_jsonable(value):
    if dataclasses.is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_config(config, save_dir, timestamp, filename="config"):
    config_dict = _to_jsonable(config)
    config_dict["EXPERIMENT_TIMESTAMP"] = timestamp
    with open(f"{save_dir}/{filename}.json", "w", encoding="utf-8") as handle:
        json.dump(config_dict, handle, indent=4, ensure_ascii=False)
