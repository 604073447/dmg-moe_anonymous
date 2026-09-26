
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os
import pandas as pd
import torch
import torch.distributed as dist

from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from Dataset.utils import prepare_ages, prepare_weights, log_weight_stats
from Utils.distributed import log_info, is_master, dist_barrier
from Utils.predefine import SPACE4


# Prepares cached arrays and wraps a DALI iterator.
def get_dali_dataloader(
    mode,
    logger,

    csv_file: str,
    img_dir: str,
    crop_size: int,

    is_training: bool,
    batch_size: int,
    num_threads: int,
    drop_last: bool,
    dali_setup: dict,

    augment_kwargs: dict,
    max_age: int,
    reweight_kwargs: dict,
    lds_kwargs: dict,
    shot_thr_dict: dict,

    ddp_info: dict,
    base_seed: int = 42,
    label_col: str = "label",
    path_col: str = "path",
):
    lds_info = (f"rw={reweight_kwargs['method']}_norm={reweight_kwargs['max_norm']}_"
                f"lds={lds_kwargs['lds']}_kn={lds_kwargs['lds_kernel']}_ks={lds_kwargs['lds_ks']}_"
                f"sig={lds_kwargs['lds_sigma']}_norm={lds_kwargs['lds_norm']}")
    data_arrays_file = csv_file.replace(".csv", f"_{lds_info}_data_arrays.npz")
    if ddp_info['rank'] == 0:
        if not os.path.exists(data_arrays_file):
            logger.info(f"Preparing data arrays from {csv_file}...")
            df = pd.read_csv(csv_file)
            paths = df[path_col].astype(str).to_numpy(dtype="U")
            ages = prepare_ages(df, label_col)
            reweight_method = reweight_kwargs.get("method", "none").lower()
            if mode == 'train' and reweight_method in ["inverse", "sqrt_inv"]:
                weights = prepare_weights(
                    df,
                    max_age=max_age,
                    label_col=label_col,
                    **reweight_kwargs,
                    **lds_kwargs,
                )
            elif mode in ['valid', 'test'] or reweight_method == "none":
                weights = np.ones((len(ages),), dtype=np.float32)
            else:
                raise ValueError(f"reweight: [{reweight_method}] must be one of: 'none', 'inverse', 'sqrt_inv'")
            np.savez(data_arrays_file, paths=paths, ages=ages, weights=weights)
            logger.info(f"Saved data arrays to: {data_arrays_file}")
    dist_barrier()

    data_arrays = np.load(data_arrays_file)
    paths = data_arrays["paths"].astype(str)
    ages = data_arrays["ages"].astype(np.float32)
    weights = data_arrays["weights"].astype(np.float32)
    dataset_size = len(ages)

    if is_master() and mode == "train":
        logger.info(f"Logging weight stats for {data_arrays_file}...")
        log_weight_stats(
            logger=logger,
            weights=weights,
            labels=ages,
            max_age=max_age,
            prefix=f"rw={reweight_kwargs['method']}_lds={lds_kwargs['lds']}",
            **shot_thr_dict,
        )


    file_list_path = data_arrays_file.replace("_data_arrays.npz", "_file_list.txt")
    if ddp_info['rank'] == 0:
        if not os.path.exists(file_list_path):
            prepare_dali_file_list(
                paths=paths,
                output_path=file_list_path,
            )
    dist_barrier()

    log_info(logger, f"{SPACE4}{mode.capitalize():<5s} samples: {dataset_size}")
    pipe = create_dali_pipeline(
        file_list=file_list_path,
        data_root=img_dir,
        crop_size=crop_size,

        shard_id=ddp_info['rank'],
        num_shards=ddp_info['world_size'],
        is_training=is_training,

        gpu_seed=base_seed+ddp_info['rank'],
        seed=base_seed+10000+ddp_info['rank'],


        batch_size=batch_size,
        num_threads=num_threads,
        device_id=ddp_info['local_rank'],

        **dali_setup,
        **augment_kwargs,
    )
    pipe.build()

    dali_iter = DALIGenericIterator(
        pipe,
        output_map=["images", "indices"],
        reader_name="Reader",
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.DROP if drop_last else LastBatchPolicy.PARTIAL,
    )

    return DALIWrapper(
        dali_iter=dali_iter,
        dataset_size=dataset_size,
        batch_size=batch_size,
        ages=ages,
        weights=weights,
        device=torch.device(f"cuda:{ddp_info['local_rank']}"),
    )


# Writes DALI file-list entries with integer sample indices.
def prepare_dali_file_list(
    paths,
    output_path: str,
):
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, path in enumerate(paths):
            rel_path = path.replace("\\", "/")
            f.write(f"{rel_path} {idx}\n")
    return


# Defines image reading, augmentation, and normalization in DALI.
@pipeline_def
def create_dali_pipeline(
    file_list: str,
    data_root: str,
    crop_size: int,

    shard_id: int,
    num_shards: int,
    is_training: bool,

    augment: bool,

    gpu_seed: int,

    _dont_use_mmap: bool = False,
    _read_ahead: bool = True,
    _prefetch_queue_depth: int = 4,
    _pad_last_batch: bool = False,

    _shuffle_after_epoch: bool = True,
    _random_shuffle: bool = False,
    _stick_to_shard: bool = False,
    _lazy_init: bool = False,

    _hw_decoder_load: float = 0.65,
    _affine: bool = True,
):

    images, indices = fn.readers.file(
        file_root=data_root,
        file_list=file_list,
        shard_id=shard_id,
        num_shards=num_shards,

        dont_use_mmap=_dont_use_mmap,
        read_ahead=_read_ahead,
        prefetch_queue_depth=_prefetch_queue_depth,
        pad_last_batch=_pad_last_batch,

        shuffle_after_epoch=(_shuffle_after_epoch and is_training),
        random_shuffle=(_random_shuffle and (not _shuffle_after_epoch) and is_training),
        stick_to_shard=(_stick_to_shard and (not _shuffle_after_epoch) and is_training),
        seed=gpu_seed,

        lazy_init=_lazy_init,
        name="Reader",
    )
    indices = fn.cast(indices, dtype=types.INT64).gpu()
    images = fn.decoders.image(
        images,
        device="mixed",
        output_type=types.RGB,
        hw_decoder_load=_hw_decoder_load,
        affine=_affine,
    )
    if augment and is_training:
        images = fn.resize(
            images,
            device="gpu",
            size=[crop_size, crop_size],
            interp_type=types.INTERP_LINEAR,
        )
        pad = 16
        canvas = crop_size + 2 * pad
        images = fn.paste(
            images,
            device="gpu",
            ratio=canvas / crop_size,
            min_canvas_size=canvas,
            fill_value=(0, 0, 0),
            paste_x=0.5,
            paste_y=0.5,
            n_channels=3,
        )
        images = fn.crop(
            images,
            device="gpu",
            crop_h=crop_size,
            crop_w=crop_size,
            crop_pos_x=fn.random.uniform(range=[0.0, 1.0]),
            crop_pos_y=fn.random.uniform(range=[0.0, 1.0]),
            rounding="truncate",
        )
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.resize(
            images,
            device="gpu",
            size=[crop_size, crop_size],
            interp_type=types.INTERP_LINEAR
        )
        mirror = 0
    images = fn.crop_mirror_normalize(
        images,
        device="gpu",
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(crop_size, crop_size),
        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
        mirror=mirror
    )
    return images, indices


# Returns DALI batches with labels and sample weights.
class DALIWrapper:
    def __init__(
        self,
        dali_iter: DALIGenericIterator,
        dataset_size: int,
        batch_size: int,
        ages: np.ndarray,
        weights: np.ndarray,
        device: torch.device,
    ):
        self.dali_iter = dali_iter
        self.dataset_size = dataset_size
        self.batch_size = batch_size

        self.all_ages = torch.from_numpy(ages).to(device=device, dtype=torch.float32, non_blocking=True)
        self.all_weights = torch.from_numpy(weights).to(device=device, dtype=torch.float32, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.dali_iter)[0]
        images = data["images"]
        indices = data["indices"].view(-1)
        return {
            "image": images,
            "label": self.all_ages[indices],
            "weight": self.all_weights[indices],
        }

    def __len__(self):
        return len(self.dali_iter)

    def reset(self):
        self.dali_iter.reset()
