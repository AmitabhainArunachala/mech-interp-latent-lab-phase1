from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.canonical.causal_state_benchmark_v2 import (
    _bootstrap_ci,
    _compute_prompt_mode_effects,
    _resolve_generation_seeds,
)
from src.pipelines.canonical.causal_state_benchmark_v1 import InterventionSpec


def test_resolve_generation_seeds_prefers_explicit_list() -> None:
    params = {"generation_seeds": [11, 22, 33]}
    assert _resolve_generation_seeds(42, params) == [11, 22, 33]


def test_bootstrap_ci_collapses_on_single_value() -> None:
    lo, hi = _bootstrap_ci([0.25], resamples=100, seed=7)
    assert lo == 0.25
    assert hi == 0.25


def test_compute_prompt_mode_effects_averages_across_seed_repeats() -> None:
    records = [
        {"prompt_id": "r1", "prompt_mode": "recursive", "condition_name": "none", "output_rv": 0.62, "bt_art": 0},
        {"prompt_id": "r1", "prompt_mode": "recursive", "condition_name": "none", "output_rv": 0.58, "bt_art": 1},
        {"prompt_id": "r1", "prompt_mode": "recursive", "condition_name": "toward_alpha_2", "output_rv": 0.49, "bt_art": 1},
        {"prompt_id": "r1", "prompt_mode": "recursive", "condition_name": "toward_alpha_2", "output_rv": 0.47, "bt_art": 1},
        {"prompt_id": "r2", "prompt_mode": "recursive", "condition_name": "none", "output_rv": 0.64, "bt_art": 0},
        {"prompt_id": "r2", "prompt_mode": "recursive", "condition_name": "none", "output_rv": 0.60, "bt_art": 0},
        {"prompt_id": "r2", "prompt_mode": "recursive", "condition_name": "toward_alpha_2", "output_rv": 0.52, "bt_art": 1},
        {"prompt_id": "r2", "prompt_mode": "recursive", "condition_name": "toward_alpha_2", "output_rv": 0.50, "bt_art": 0},
        {"prompt_id": "b1", "prompt_mode": "baseline", "condition_name": "none", "output_rv": 0.70, "bt_art": 0},
        {"prompt_id": "b1", "prompt_mode": "baseline", "condition_name": "toward_alpha_2", "output_rv": 0.69, "bt_art": 0},
    ]

    effects = _compute_prompt_mode_effects(
        records,
        interventions=[
            InterventionSpec(name="none", alpha=0.0),
            InterventionSpec(name="toward_alpha_2", alpha=2.0),
        ],
        control_name="none",
        prompt_mode="recursive",
        bootstrap_resamples=200,
        seed=3,
    )

    effect = effects["toward_alpha_2"]
    assert effect["n_prompt_pairs"] == 2
    assert effect["rv_delta_mean"] < 0.0
    assert effect["bt_art_rate_delta"] > 0.0
    assert effect["rv_delta_ci_95"][1] < 0.0
