"""
Attribution module for Futures-Six.

Provides portfolio-consistent sleeve-level return attribution.
"""

from src.attribution.core import (
    compute_attribution,
    decompose_weights_by_sleeve,
)
from src.attribution.artifacts import generate_attribution_artifacts

__all__ = [
    "compute_attribution",
    "decompose_weights_by_sleeve",
    "generate_attribution_artifacts",
]
