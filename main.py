import time
import torch

from App.bootstrap import build_config, build_loader, build_model
from App.cli import parse_args
from App.setup import setup_experiment
from Utils.distributed import log_info, cleanup_distributed, dist_barrier
from Phase.run_init import run_init

torch.set_float32_matmul_precision("high")


# Runs the AgeDB-DIR training workflow.
def main():
    start_time = time.time()
    args = parse_args()
    timestamp, exp_dir, ddp_info, logger, device = setup_experiment(args)
    log_info(logger, "Building Configuration...")
    config = build_config(logger, args, exp_dir, timestamp)
    log_info(logger, "Building Dataloader...")
    train_sampler, train_loader = build_loader("train", config, logger, ddp_info)
    _, valid_loader = build_loader("valid", config, logger, ddp_info)
    _, test_loader = build_loader("test", config, logger, ddp_info)
    log_info(logger, "Building Model...")
    model = build_model(config, device, ddp_info)
    log_info(logger, "Run Phase [Init]")
    run_init(
        config, device, model, ddp_info,
        train_loader, train_sampler, valid_loader, test_loader,
        logger,
    )
    dist_barrier()
    log_info(logger, f"Experiment complete. Elapsed time: {time.time() - start_time:.2f} seconds")
    cleanup_distributed()


if __name__ == "__main__":
    main()
