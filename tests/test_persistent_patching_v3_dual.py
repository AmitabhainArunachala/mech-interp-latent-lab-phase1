from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.persistent_patching_summary import (
    PERSISTED_AGGREGATE_KEYS,
    aggregate_sessions,
    serialize_aggregates,
)


def _make_session(session_id: str, turns: list[dict]) -> dict:
    bt_art_count = sum(
        turn["classification"] in {"BREAKTHROUGH", "ARTICULATE"} for turn in turns
    )
    rv_values = [turn["output_rv"] for turn in turns if turn["output_rv"] is not None]
    return {
        "session_id": session_id,
        "max_turns": len(turns),
        "bt_art_count": bt_art_count,
        "bt_art_rate": bt_art_count / len(turns),
        "mean_rv": float(np.mean(rv_values)) if rv_values else None,
        "std_rv": float(np.std(rv_values)) if rv_values else None,
        "classification_dist": dict(Counter(turn["classification"] for turn in turns)),
        "turns": turns,
    }


def test_serialized_aggregates_preserve_quality_summary_fields() -> None:
    session_a = _make_session(
        "recursive_clean_0",
        [
            {"output_rv": 0.4, "alpha_ratio": 0.8, "classification": "BREAKTHROUGH"},
            {"output_rv": None, "alpha_ratio": 0.2, "classification": "MALFORMED"},
            {"output_rv": 0.9, "alpha_ratio": 0.6, "classification": "REPETITIVE"},
        ],
    )
    session_b = _make_session(
        "recursive_clean_1",
        [
            {"output_rv": 0.5, "alpha_ratio": 1.0, "classification": "ARTICULATE"},
            {"output_rv": 0.7, "alpha_ratio": 0.4, "classification": "REPETITIVE"},
        ],
    )

    aggregate = aggregate_sessions([session_a, session_b])
    serialized = serialize_aggregates({"recursive_clean": aggregate})
    saved = serialized["recursive_clean"]

    assert set(PERSISTED_AGGREGATE_KEYS).issubset(saved)
    assert "per_session" not in saved
    assert saved["total_malformed"] == 1
    assert saved["malformed_rate"] == pytest.approx(0.2)
    assert saved["total_repetitive"] == 2
    assert saved["repetitive_rate"] == pytest.approx(0.4)
    assert saved["mean_alpha_ratio"] == pytest.approx(0.6)
    assert saved["n_rv"] == 4
    assert saved["n_rv_missing"] == 1
