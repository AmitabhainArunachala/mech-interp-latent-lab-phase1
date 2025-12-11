"""
Core utilities: model loading, type definitions, and standardized hook context managers.
"""

from .models import load_model, set_seed
from .hooks import capture_v_projection, capture_hidden_states
from .utils import behavior_score

__all__ = [
    "load_model",
    "set_seed",
    "capture_v_projection",
    "capture_hidden_states",
    "behavior_score",
]

