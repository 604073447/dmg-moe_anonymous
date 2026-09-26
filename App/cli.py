import argparse


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNetDMGMoE on AgeDB-DIR")
    parser.add_argument("--exp_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--lr_scale", type=str, choices=["none", "linear", "sqrt"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)
