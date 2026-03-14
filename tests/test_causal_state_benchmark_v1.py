from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompts.loader import PromptLoader
from src.pipelines.canonical.causal_state_benchmark_v1 import (
    _build_blind_packet,
    _build_holdout_prompt_set,
    _paired_effects,
    _select_source_records,
    InterventionSpec,
)


def _make_record(
    session_id: str,
    session_type: str,
    turn: int,
    classification: str,
    output_rv: float,
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "session_type": session_type,
        "turn": turn,
        "classification": classification,
        "output_rv": output_rv,
        "response": f"response {session_id} {turn} " + ("x" * 120),
        "path": f"{session_id}.json",
    }


def test_select_source_records_respects_session_caps() -> None:
    records = [
        _make_record("rec_a", "recursive", 0, "BREAKTHROUGH", 0.31),
        _make_record("rec_a", "recursive", 1, "ARTICULATE", 0.35),
        _make_record("rec_b", "recursive", 0, "ARTICULATE", 0.29),
        _make_record("rec_b", "recursive", 1, "ARTICULATE", 0.52),
        _make_record("base_a", "baseline", 0, "SURFACE", 0.88),
        _make_record("base_a", "baseline", 1, "REPETITIVE", 0.82),
        _make_record("base_b", "baseline", 0, "SURFACE", 0.91),
        _make_record("base_b", "baseline", 1, "SURFACE", 0.67),
    ]

    selected = _select_source_records(
        records,
        positive_classes={"BREAKTHROUGH", "ARTICULATE"},
        negative_classes={"SURFACE", "REPETITIVE"},
        positive_quantile=0.5,
        negative_quantile=0.5,
        positive_session_types={"recursive"},
        negative_session_types={"baseline"},
        max_source_per_label=3,
        max_source_per_session=1,
        seed=7,
    )

    positive = selected["positive_records"]
    negative = selected["negative_records"]

    assert len(positive) == 2
    assert len(negative) == 2
    assert {row["session_id"] for row in positive} == {"rec_a", "rec_b"}
    assert {row["session_id"] for row in negative} == {"base_a", "base_b"}
    assert all(float(row["output_rv"]) <= selected["positive_threshold"] for row in positive)
    assert all(float(row["output_rv"]) >= selected["negative_threshold"] for row in negative)


def test_holdout_prompt_set_is_balanced_and_reproducible() -> None:
    loader = PromptLoader()
    recursive_groups = ["L3_deeper", "L4_full"]
    baseline_groups = ["baseline_math", "baseline_factual"]

    holdout_a = _build_holdout_prompt_set(
        loader,
        recursive_groups=recursive_groups,
        baseline_groups=baseline_groups,
        holdout_per_group=3,
        seed=11,
    )
    holdout_b = _build_holdout_prompt_set(
        loader,
        recursive_groups=recursive_groups,
        baseline_groups=baseline_groups,
        holdout_per_group=3,
        seed=11,
    )

    assert holdout_a == holdout_b
    assert len(holdout_a) == 12
    counts = {}
    for row in holdout_a:
        counts[row["prompt_group"]] = counts.get(row["prompt_group"], 0) + 1
    assert counts == {
        "L3_deeper": 3,
        "L4_full": 3,
        "baseline_factual": 3,
        "baseline_math": 3,
    }


def test_build_blind_packet_hides_condition_labels(tmp_path: Path) -> None:
    rows = [
        {
            "prompt_id": "p1",
            "prompt_mode": "baseline",
            "prompt_group": "baseline_math",
            "prompt_text": "Prompt one",
            "generated_text": "Response one",
            "condition_name": "none",
            "alpha": 0.0,
            "classification": "SURFACE",
            "bt_art": 0,
        },
        {
            "prompt_id": "p1",
            "prompt_mode": "baseline",
            "prompt_group": "baseline_math",
            "prompt_text": "Prompt one",
            "generated_text": "Response two",
            "condition_name": "toward_low_rv",
            "alpha": 2.0,
            "classification": "ARTICULATE",
            "bt_art": 1,
        },
    ]

    blind_csv = tmp_path / "blind.csv"
    blind_key = tmp_path / "blind_key.json"
    _build_blind_packet(rows, seed=3, csv_path=blind_csv, key_path=blind_key)

    with blind_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)

    key_rows = json.loads(blind_key.read_text())

    assert len(csv_rows) == 2
    assert "condition_name" not in csv_rows[0]
    assert "alpha" not in csv_rows[0]
    assert {row["sample_id"] for row in csv_rows} == {row["sample_id"] for row in key_rows}
    assert {row["condition_name"] for row in key_rows} == {"none", "toward_low_rv"}


def test_paired_effects_reports_expected_signs() -> None:
    records = [
        {
            "prompt_id": "a",
            "prompt_mode": "baseline",
            "condition_name": "none",
            "output_rv": 0.72,
            "bt_art": 0,
        },
        {
            "prompt_id": "a",
            "prompt_mode": "baseline",
            "condition_name": "toward_low_rv",
            "output_rv": 0.51,
            "bt_art": 1,
        },
        {
            "prompt_id": "a",
            "prompt_mode": "baseline",
            "condition_name": "away_from_low_rv",
            "output_rv": 0.84,
            "bt_art": 0,
        },
        {
            "prompt_id": "b",
            "prompt_mode": "recursive",
            "condition_name": "none",
            "output_rv": 0.58,
            "bt_art": 1,
        },
        {
            "prompt_id": "b",
            "prompt_mode": "recursive",
            "condition_name": "toward_low_rv",
            "output_rv": 0.43,
            "bt_art": 1,
        },
        {
            "prompt_id": "b",
            "prompt_mode": "recursive",
            "condition_name": "away_from_low_rv",
            "output_rv": 0.70,
            "bt_art": 0,
        },
    ]

    effects = _paired_effects(
        records,
        control_name="none",
        interventions=[
            InterventionSpec(name="none", alpha=0.0),
            InterventionSpec(name="toward_low_rv", alpha=2.0),
            InterventionSpec(name="away_from_low_rv", alpha=-2.0),
        ],
    )

    assert effects["toward_low_rv"]["rv_delta_mean"] < 0.0
    assert effects["toward_low_rv"]["bt_art_rate_delta"] > 0.0
    assert effects["away_from_low_rv"]["rv_delta_mean"] > 0.0
    assert effects["away_from_low_rv"]["bt_art_rate_delta"] < 0.0
