import torch

from Utils.distributed import is_master


def save_checkpoint(save_path: str, model, record: dict, logger):
    if is_master():
        checkpoint = {
            'model': model,
            'record': record,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}.")
