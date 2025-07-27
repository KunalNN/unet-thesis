
"""Compatibility layer for legacy imports.

The implementation of metric functions lives in :mod:`src.metrics`.  This file
simply re-exports those functions so that older modules importing
``src.utils.metrics`` remain functional.
"""

from src.metrics import compute_psnr, compute_ssim, compute_lpips

__all__ = ["compute_psnr", "compute_ssim", "compute_lpips"]
