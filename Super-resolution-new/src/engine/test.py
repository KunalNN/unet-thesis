# src/engine/test.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.datasets.div2k import get_set14_local
from src.utils.metrics import compute_psnr, compute_ssim, compute_lpips

def main(args):
    # 1) load model
    model = load_model(args.checkpoint, compile=False)

    # 2) dataset
    ds = get_set14_local(args.data_root, scale=args.scale, batch_size=1)

    # 3) metrics
    results = []
    for lr, hr in ds:
        sr = model.predict(lr)
        results.append({
            "psnr": compute_psnr(hr, sr),
            "ssim": compute_ssim(hr, sr),
            "lpips": compute_lpips(hr, sr),
        })

    # 4) save CSV
    import csv
    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, "set14_results.csv")
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["psnr","ssim","lpips"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results → {outpath}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",   type=str, required=True)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--scale",       type=int, default=4)
    p.add_argument("--output-dir",  type=str, default="outputs/results")
    args = p.parse_args()
    main(args)
