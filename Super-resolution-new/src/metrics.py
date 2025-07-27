# src/utils/metrics.py
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import lpips_tf
import numpy as np

def compute_psnr(hr, sr):
    """Batches of images in [0,1]."""
    hr_np = hr.numpy()
    sr_np = sr.numpy()
    return float(np.mean([sk_psnr(h, s, data_range=1.0)
                          for h, s in zip(hr_np, sr_np)]))

def compute_ssim(hr, sr):
    hr_np = hr.numpy()
    sr_np = sr.numpy()
    return float(np.mean([sk_ssim(h, s, multichannel=True, data_range=1.0)
                          for h, s in zip(hr_np, sr_np)]))

_lpips = lpips_tf.LPIPS(net='vgg')  # initialize once

def compute_lpips(hr, sr):
    """hr, sr: tf.Tensor [B,H,W,3] in [0,1]"""
    # LPIPS expects [-1,1]
    hr_in = hr * 2 - 1
    sr_in = sr * 2 - 1
    val = _lpips(hr_in, sr_in)
    return float(tf.reduce_mean(val).numpy())
