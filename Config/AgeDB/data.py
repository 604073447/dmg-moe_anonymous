from dataclasses import dataclass, field


@dataclass
class DataConfig:
    train_csv: str = "Data/AgeDB/meta/train.csv"
    valid_csv: str = "Data/AgeDB/meta/valid.csv"
    test_csv: str = "Data/AgeDB/meta/test.csv"
    img_dir: str = "/dev/shm/AgeDB_cache"
    label_col: str = "label"
    path_col: str = "path"
    img_info: dict = field(default_factory=lambda: {"channels": 3, "width": 224, "height": 224})
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 16
    drop_last: bool = True
    dali_setup: dict = field(default_factory=lambda: {
        "_dont_use_mmap": False,
        "_read_ahead": True,
        "_prefetch_queue_depth": 4,
        "_pad_last_batch": False,
        "_shuffle_after_epoch": True,
        "_random_shuffle": False,
        "_stick_to_shard": False,
        "_lazy_init": False,
        "_hw_decoder_load": 0.65,
        "_affine": True,
    })
    augment: dict = field(default_factory=lambda: {"augment": True})
    max_age: int = 121
    reweight: dict = field(default_factory=lambda: {"method": "sqrt_inv", "max_norm": None})
    lds: dict = field(default_factory=lambda: {
        "lds": True,
        "lds_kernel": "gaussian",
        "lds_ks": 5,
        "lds_sigma": 2.0,
        "lds_norm": "sum",
    })
    eval_bins: dict = field(default_factory=lambda: {"max_age": 121, "bin_step": 20, "num_bins": None})
    shot_thr_dict: dict = field(default_factory=lambda: {"many_shot_thr": 100, "low_shot_thr": 20})
    label_map_dict: dict = field(default_factory=lambda: {
        "bins": None,
        "label_bin_map": None,
        "bin_name_map": None,
        "label_shot_map": None,
        "shot_name_map": None,
    })
