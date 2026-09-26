# DMG-MoE

This repository contains the DMG-MoE implementation for age regression on AgeDB-DIR. The model combines predictions from multiple ResNet depths using learned gates.

## Requirements

The training pipeline uses PyTorch and NVIDIA DALI on an NVIDIA GPU. Install the pinned Python dependencies with:

```bash
pip install -r requirements_cloud.txt
```

The requirements file specifies the CUDA 13.0 build of DALI. Use a compatible CUDA environment.

## Data

The repository includes the train, validation, and test metadata files under `Data/AgeDB/meta/`. Image files are not included. Obtain the AgeDB images separately under their applicable terms.

Each CSV has `label`, `path`, and `split` columns. The `path` values are relative to the image root. Set `DataConfig.img_dir` in `Config/AgeDB/data.py` to that root; its default is `/dev/shm/AgeDB_cache`. The directory containing the CSV files must be writable because the data loader creates cache arrays and DALI file lists there.

## Training

From the repository root, run:

```bash
python main.py --exp_dir Results/example --seed 42
```

Optional command-line arguments include `--batch_size`, `--num_workers`, and `--lr_scale`. The default configuration is defined in `Config/AgeDB/`.

The run writes `config.json`, `AgeDB.log`, model files, and a checkpoint to the experiment directory. Validation and test metrics are logged during training. The best model is selected using validation MAE for all samples.
