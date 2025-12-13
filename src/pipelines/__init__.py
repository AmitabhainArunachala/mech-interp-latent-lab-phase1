"""
High-level experiment orchestrators (Phase 1, Phase 2, etc.).
"""

from .phase1_existence import run_phase1_existence_proof
from .registry import run_from_config

__all__ = [
    "run_phase1_existence_proof",
    "run_from_config",
]

