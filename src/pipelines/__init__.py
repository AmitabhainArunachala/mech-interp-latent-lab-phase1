"""
Experiment pipelines organized in three tiers:
- canonical/: Core paper findings (7 pipelines)
- discovery/: Methodology tools for new models (12 pipelines)
- archive/: Historical/superseded code (35 pipelines)
"""

from .registry import run_from_config

__all__ = [
    "run_from_config",
]


def run_phase1_existence_proof(*args, **kwargs):
    """Legacy export for backwards compatibility."""
    from .archive.phase1_existence import run_phase1_existence_proof as _run
    return _run(*args, **kwargs)

