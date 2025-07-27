
"""Compatibility layer for legacy imports.

The visualization helpers were moved to :mod:`src.viz`.  This module simply
re-exports them so that imports from ``src.utils.viz`` continue to work.
"""

from src.viz import plot_sr_triplet

__all__ = ["plot_sr_triplet"]
