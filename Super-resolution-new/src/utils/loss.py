
"""Compatibility layer for legacy imports.

This module re-exports the loss functions defined in :mod:`src.loss` so that
existing code that imports from ``src.utils.loss`` continues to work even
after the refactor that moved these utilities to the top-level ``src``
package.
"""

from src.loss import charbonnier_loss, l1_loss

__all__ = ["charbonnier_loss", "l1_loss"]
