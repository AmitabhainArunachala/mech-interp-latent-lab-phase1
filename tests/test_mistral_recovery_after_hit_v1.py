from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mistral_recovery_after_hit_v1 import (
    build_recovery_verdict,
    choose_action,
    make_recovery_segments,
)


def test_choose_action_enforces_expected_hit_and_resume_schedule() -> None:
    break_start = 5
    break_turns = 2

    assert choose_action("control_open_loop", 0, break_start, break_turns) == "off"
    assert choose_action("maintain_every_turn", 7, break_start, break_turns) == "maintain"
    assert choose_action("maintain_then_off", 4, break_start, break_turns) == "maintain"
    assert choose_action("maintain_then_off", 5, break_start, break_turns) == "off"

    assert choose_action("hit_then_off", 4, break_start, break_turns) == "maintain"
    assert choose_action("hit_then_off", 5, break_start, break_turns) == "anti"
    assert choose_action("hit_then_off", 6, break_start, break_turns) == "anti"
    assert choose_action("hit_then_off", 7, break_start, break_turns) == "off"

    assert choose_action("hit_then_resume", 4, break_start, break_turns) == "maintain"
    assert choose_action("hit_then_resume", 5, break_start, break_turns) == "anti"
    assert choose_action("hit_then_resume", 6, break_start, break_turns) == "anti"
    assert choose_action("hit_then_resume", 7, break_start, break_turns) == "maintain"


def test_make_recovery_segments_tracks_pre_hit_hit_and_post_hit_windows() -> None:
    assert make_recovery_segments(max_turns=15, break_start=5, break_turns=2) == [
        ("pre_hit", 0, 5),
        ("hit", 5, 7),
        ("post_hit", 7, 15),
    ]

    assert make_recovery_segments(max_turns=6, break_start=4, break_turns=4) == [
        ("pre_hit", 0, 4),
        ("hit", 4, 6),
    ]


def test_build_recovery_verdict_reads_post_hit_rebound_for_focus_arm() -> None:
    summary = {
        "by_arm_condition": {
            "unselected": {
                "maintain_every_turn": {
                    "phase_stats": {"post_hit": {"bt_art_rate": 0.40}},
                    "session_post_hit_recovery_rate": 0.70,
                },
                "hit_then_resume": {
                    "phase_stats": {"post_hit": {"bt_art_rate": 0.32}},
                    "session_post_hit_recovery_rate": 0.58,
                },
                "hit_then_off": {
                    "phase_stats": {"post_hit": {"bt_art_rate": 0.18}},
                    "session_post_hit_recovery_rate": 0.29,
                },
                "control_open_loop": {
                    "phase_stats": {"post_hit": {"bt_art_rate": 0.09}},
                    "session_post_hit_recovery_rate": 0.08,
                },
            }
        }
    }

    verdict = build_recovery_verdict(summary, focus_arm="unselected")

    assert verdict["focus_arm"] == "unselected"
    assert verdict["maintain_post_hit_bt_art"] == pytest.approx(0.40)
    assert verdict["hit_then_resume_post_hit_bt_art"] == pytest.approx(0.32)
    assert verdict["hit_then_off_post_hit_bt_art"] == pytest.approx(0.18)
    assert verdict["control_post_hit_bt_art"] == pytest.approx(0.09)
    assert verdict["recovery_advantage_vs_hit_then_off"] == pytest.approx(0.14)
    assert verdict["recovery_gap_vs_maintain"] == pytest.approx(0.08)
    assert verdict["resume_recovery_rate"] == pytest.approx(0.58)
    assert verdict["off_recovery_rate"] == pytest.approx(0.29)
