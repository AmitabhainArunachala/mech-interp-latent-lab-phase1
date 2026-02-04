"""
rv_toolkit - Representational Volume metrics for transformer interpretability

Measuring geometric signatures of recursive self-reference in transformer value spaces.

Key metrics:
- R_V (Representational Volume): Participation ratio measuring effective dimensionality
- Dual-space decomposition: V_parallel and V_perpendicular components
- Geometric homeostasis: Cross-layer compensation patterns

Paper: "Coordinated Dual-Space Geometric Transformations Mediate Recursive 
       Self-Reference in Transformer Value Spaces"

Usage:
    from rv_toolkit import compute_rv, ActivationPatcher
    
    # Compute R_V for a value tensor
    rv = compute_rv(v_tensor, window_size=16)
    
    # Run activation patching experiment
    patcher = ActivationPatcher(model, tokenizer)
    results = patcher.run_experiment(baseline_prompts, recursive_prompts)
"""

__version__ = "0.1.0"

from .metrics import (
    compute_rv,
    compute_participation_ratio,
    compute_effective_rank,
    compute_dual_space_decomposition,
    RVResult,
)

from .patching import (
    ActivationPatcher,
    PatchingResult,
    ControlCondition,
)

from .analysis import (
    compute_transfer_efficiency,
    compute_effect_size,
    run_statistical_tests,
    AnalysisResult,
)

from .prompts import (
    RECURSIVE_PROMPTS,
    BASELINE_PROMPTS,
    get_prompt_pairs,
)

__all__ = [
    # Metrics
    "compute_rv",
    "compute_participation_ratio", 
    "compute_effective_rank",
    "compute_dual_space_decomposition",
    "RVResult",
    # Patching
    "ActivationPatcher",
    "PatchingResult",
    "ControlCondition",
    # Analysis
    "compute_transfer_efficiency",
    "compute_effect_size",
    "run_statistical_tests",
    "AnalysisResult",
    # Prompts
    "RECURSIVE_PROMPTS",
    "BASELINE_PROMPTS",
    "get_prompt_pairs",
]
