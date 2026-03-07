"""
GeometricLens: A toolkit for measuring geometric signatures of computation in transformers.

Developed during mechanistic interpretability research on R_V contraction
in V-projection space during recursive self-referential processing.

Usage:
    from geometric_lens import GeometricProbe

    probe = GeometricProbe("mistralai/Mistral-7B-v0.1")
    result = probe.measure("What is the nature of self-reference?")
    print(result.rv, result.spectral_gap, result.mode)
"""

from .probe import GeometricProbe, GeometricResult
from .metrics import (
    participation_ratio,
    compute_rv,
    compute_rv_with_components,
    compute_spectral_stats,
    compute_cosine_similarity,
    compute_attention_entropy,
    SpectralStats,
)
from .hooks import (
    capture_v_projection,
    capture_k_projection,
    capture_q_projection,
    capture_attention_patterns,
    capture_hidden_states,
)
from .models import ModelRegistry, ModelSpec

__version__ = "0.1.0"
__all__ = [
    "GeometricProbe",
    "GeometricResult",
    "participation_ratio",
    "compute_rv",
    "compute_rv_with_components",
    "compute_spectral_stats",
    "compute_cosine_similarity",
    "compute_attention_entropy",
    "SpectralStats",
    "capture_v_projection",
    "capture_k_projection",
    "capture_q_projection",
    "capture_attention_patterns",
    "capture_hidden_states",
    "ModelRegistry",
    "ModelSpec",
]
