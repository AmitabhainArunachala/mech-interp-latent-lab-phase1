"""
Metrics module: R_V calculation, participation ratio, and SVD utilities.

The Measurement Invariant:
- Always measure R_V on the prompt tokens (last W=16), not generated tokens.
- Always use torch.linalg.svd(..., full_matrices=False) and handle degenerate singular values.
"""

from .rv import compute_rv, participation_ratio

__all__ = [
    "compute_rv",
    "participation_ratio",
]

