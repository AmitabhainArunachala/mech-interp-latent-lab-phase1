"""
Metrics module: R_V calculation, participation ratio, SVD utilities, and extended metrics.

The Measurement Invariant:
- Always measure R_V on the prompt tokens (last W=16), not generated tokens.
- Always use torch.linalg.svd(..., full_matrices=False) and handle degenerate singular values.

Core metrics:
- R_V (PR ratio) - geometric contraction
- Participation Ratio - effective dimensionality

Extended metrics (publication-grade):
- Cosine Similarity - directional alignment early/late
- Spectral Stats - top-1 ratio, spectral gap, effective rank
- Attention Entropy - focus at readout layer
"""

from .rv import compute_rv, participation_ratio
from .extended import (
    compute_cosine_similarity,
    compute_spectral_stats,
    compute_attention_entropy,
    compute_extended_metrics,
    compute_extended_batch,
    ExtendedMetrics,
    SpectralStats,
)

__all__ = [
    # Core
    "compute_rv",
    "participation_ratio",
    # Extended
    "compute_cosine_similarity",
    "compute_spectral_stats",
    "compute_attention_entropy",
    "compute_extended_metrics",
    "compute_extended_batch",
    "ExtendedMetrics",
    "SpectralStats",
]

