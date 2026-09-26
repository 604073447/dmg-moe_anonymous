import os
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel


def setup_distributed(backend='nccl'):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, device_id=torch.device(f'cuda:{local_rank}'))
        return {
            'rank': rank,
            'world_size': world_size,
            'local_rank': local_rank
        }
    else:
        return {'rank' : 0,'world_size' : 1,'local_rank' : 0}

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0

def dist_barrier():
    if dist.is_initialized():
        dist.barrier()

def replace_sync_bn(model):
    if dist.is_initialized() and dist.get_world_size() > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model

def log_info(logger, message=""):
    if is_master():
        logger.info(message)

def wrap_model(model, ddp_info, find_unused_parameters=False):
    if dist.is_initialized():
        local_rank = ddp_info['local_rank']
        model = model.to(f"cuda:{local_rank}")
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
            bucket_cap_mb=300,
        )
    return model

def unwrap_model(model):
    return model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model

def gather_dict(
    source_dict: Dict[str, Any],
    average: bool = False,
    weight: Optional[float] = None
) -> Dict[str, Any]:
    if not dist.is_initialized():
        return source_dict
    world_size = dist.get_world_size()

    tensors_to_gather = {k: v for k, v in source_dict.items() if isinstance(v, torch.Tensor)}
    objects_to_gather = {k: v for k, v in source_dict.items() if not isinstance(v, torch.Tensor)}

    merged_dict: Dict[str, Any] = {}
    if objects_to_gather:
        gathered_objs = [None] * world_size
        dist.all_gather_object(gathered_objs, objects_to_gather)

        for k in objects_to_gather.keys():
            vals = [d[k] for d in gathered_objs]
            if average:
                w_list = [1.0] * world_size
                if weight is not None:
                    dist.all_gather_object(w_list, weight)
                merged_dict[k] = np.average(vals, axis=0, weights=w_list)
            else:
                merged_dict[k] = vals

    for k, v in tensors_to_gather.items():
        merged_dict[k] = gather_variable_length_tensor(v, concat=True)

    return merged_dict

def gather_variable_length_tensor(tensor, concat=False):
    if not dist.is_initialized():
        return tensor if concat else [tensor]

    local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
    world_size = dist.get_world_size()
    sizes_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes_list, local_size)
    sizes = torch.cat(sizes_list)

    pad_length = sizes.max().item() - tensor.shape[0]
    if pad_length > 0:
        pad_size = [pad_length] + list(tensor.shape[1:])
        padded = torch.cat([tensor, torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)], dim=0)
    else:
        padded = tensor

    tensor_list = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(tensor_list, padded)

    out = [t[:size] for t, size in zip(tensor_list, sizes)]
    if concat:
        out = torch.cat(out, dim=0)
    return out
