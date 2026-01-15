"""
Core utilities: model loading, type definitions, and standardized hook context managers.
"""

from .models import load_model, set_seed
from .hooks import capture_v_projection, capture_hidden_states
from .patching import (
    PersistentVPatcher,
    PersistentResidualPatcher,
    extract_v_activation,
    extract_residual_activation,
)
from .utils import behavior_score

__all__ = [
    "load_model",
    "set_seed",
    "capture_v_projection",
    "PersistentVPatcher",
    "PersistentResidualPatcher",
    "extract_v_activation",
    "extract_residual_activation",
    "capture_hidden_states",
    "behavior_score",
]

