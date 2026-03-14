from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompts.loader import PromptLoader
from src.pipelines.canonical.causal_state_targeted_scan_v1 import (
    _build_scan_prompt_set,
    _candidate_objective,
    _resolve_scan_candidates,
)


def test_resolve_scan_candidates_builds_full_grid() -> None:
    candidates = _resolve_scan_candidates(
        {
            "candidate_source_layers": [25, 27],
            "candidate_windows": [8, 16],
            "candidate_alpha_magnitudes": [2.0],
        }
    )

    assert [candidate.name for candidate in candidates] == [
        "L25_W8_A2",
        "L25_W16_A2",
        "L27_W8_A2",
        "L27_W16_A2",
    ]


def test_build_scan_prompt_set_balances_modes_and_groups() -> None:
    loader = PromptLoader()
    prompts = _build_scan_prompt_set(
        loader,
        recursive_groups=["L3_deeper", "L4_full"],
        baseline_groups=["baseline_math", "baseline_factual"],
        recursive_per_group=2,
        baseline_per_group=3,
        seed=9,
    )

    counts_by_group: dict[str, int] = {}
    counts_by_mode: dict[str, int] = {}
    for row in prompts:
        counts_by_group[row["prompt_group"]] = counts_by_group.get(row["prompt_group"], 0) + 1
        counts_by_mode[row["prompt_mode"]] = counts_by_mode.get(row["prompt_mode"], 0) + 1

    assert counts_by_group == {
        "L3_deeper": 2,
        "L4_full": 2,
        "baseline_factual": 3,
        "baseline_math": 3,
    }
    assert counts_by_mode == {"baseline": 6, "recursive": 4}


def test_candidate_objective_prefers_recursive_gain_with_specificity() -> None:
    stronger = _candidate_objective(
        recursive_effects={
            "toward": {"bt_art_rate_delta": 0.18, "rv_delta_mean": -0.03},
            "away": {"bt_art_rate_delta": -0.12, "rv_delta_mean": 0.02},
        },
        baseline_effects={
            "toward": {"bt_art_rate_delta": 0.01, "rv_delta_mean": 0.00},
            "away": {"bt_art_rate_delta": 0.00, "rv_delta_mean": 0.00},
        },
    )
    weaker = _candidate_objective(
        recursive_effects={
            "toward": {"bt_art_rate_delta": 0.06, "rv_delta_mean": -0.01},
            "away": {"bt_art_rate_delta": 0.02, "rv_delta_mean": -0.01},
        },
        baseline_effects={
            "toward": {"bt_art_rate_delta": 0.09, "rv_delta_mean": 0.03},
            "away": {"bt_art_rate_delta": 0.05, "rv_delta_mean": 0.00},
        },
    )

    assert stronger["score"] > weaker["score"]
    assert stronger["sign_match_count"] == 4
    assert weaker["sign_checks"]["recursive_away_bt_negative"] is False
