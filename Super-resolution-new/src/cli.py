# src/cli.py
import argparse
from src.engine.train import main as train_main
from src.engine.test import main as test_main

def main():
    p = argparse.ArgumentParser(prog="sr_cli")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("train")
    for arg in ["data_root","scale","crop","batch_size","epochs","base_filters","checkpoint_dir","log_dir"]:
        sp.add_argument(f"--{arg.replace('_','-')}", required=True if arg=="data_root" else False)

    sp = sub.add_parser("test")
    sp.add_argument("--data-root",  required=True)
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--scale",      default=4)
    sp.add_argument("--output-dir", default="outputs/results")

    args = p.parse_args()
    if args.command=="train":
        train_main(args)
    elif args.command=="test":
        test_main(args)

if __name__=="__main__":
    main()
