from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.mistral_program import (
    effective_active_leases,
    reconcile_program_registry,
)


def test_reconcile_program_registry_completes_queue_when_result_is_newer_than_running_lease(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "results" / "queue_a" / "run_01" / "summary.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("{}", encoding="utf-8")

    registry = {
        "experiments": [
            {
                "experiment_id": "exp_a",
                "queue_group": "queue_a",
                "status": "queued",
                "artifact_glob": "results/queue_a/*/summary.json",
            }
        ],
        "queue_units": [
            {
                "queue_unit_id": "queue_a",
                "queue_group": "queue_a",
                "status": "queued",
                "experiment_ids": ["exp_a"],
                "priority": 100,
            }
        ],
    }
    results = {
        "updated_at": "2026-03-19T12:00:00Z",
        "results": [
            {
                "experiment_id": "exp_a",
                "queue_group": "queue_a",
                "status": "completed",
                "artifact_path": "results/queue_a/run_01/summary.json",
                "updated_at": "2026-03-19T12:00:00Z",
                "run_id": "run_01",
            }
        ],
    }
    leases = {
        "updated_at": "2026-03-19T10:00:00Z",
        "leases": [
            {
                "pod_name": "pod-a",
                "queue_group": "queue_a",
                "status": "running",
                "run_id": "run_old",
                "current_step": "exp_a",
                "updated_at": "2026-03-19T10:00:00Z",
            }
        ],
    }

    reconciled = reconcile_program_registry(
        registry,
        results_payload=results,
        leases_payload=leases,
        repo_root=tmp_path,
    )

    exp = reconciled["experiments"][0]
    unit = reconciled["queue_units"][0]

    assert exp["status"] == "completed"
    assert "active_lease" not in exp
    assert unit["status"] == "completed"
    assert "active_lease" not in unit


def test_effective_active_leases_ignores_completed_queue_with_stale_running_lease(
    tmp_path: Path,
) -> None:
    completed_artifact = tmp_path / "results" / "queue_a" / "run_01" / "summary.json"
    completed_artifact.parent.mkdir(parents=True, exist_ok=True)
    completed_artifact.write_text("{}", encoding="utf-8")

    live_artifact = tmp_path / "results" / "queue_b" / "run_02" / "partial.json"
    live_artifact.parent.mkdir(parents=True, exist_ok=True)
    live_artifact.write_text("{}", encoding="utf-8")

    registry = {
        "experiments": [
            {
                "experiment_id": "exp_a",
                "queue_group": "queue_a",
                "status": "queued",
                "artifact_glob": "results/queue_a/*/summary.json",
            },
            {
                "experiment_id": "exp_b",
                "queue_group": "queue_b",
                "status": "running",
                "artifact_glob": "results/queue_b/*/summary.json",
            },
        ],
        "queue_units": [
            {
                "queue_unit_id": "queue_a",
                "queue_group": "queue_a",
                "status": "queued",
                "experiment_ids": ["exp_a"],
                "priority": 100,
            },
            {
                "queue_unit_id": "queue_b",
                "queue_group": "queue_b",
                "status": "queued",
                "experiment_ids": ["exp_b"],
                "priority": 200,
            },
        ],
    }
    results = {
        "updated_at": "2026-03-19T12:00:00Z",
        "results": [
            {
                "experiment_id": "exp_a",
                "queue_group": "queue_a",
                "status": "completed",
                "artifact_path": "results/queue_a/run_01/summary.json",
                "updated_at": "2026-03-19T12:00:00Z",
                "run_id": "run_01",
            }
        ],
    }
    leases = {
        "updated_at": "2026-03-19T12:05:00Z",
        "leases": [
            {
                "pod_name": "pod-a",
                "queue_group": "queue_a",
                "status": "running",
                "run_id": "run_old",
                "current_step": "exp_a",
                "updated_at": "2026-03-19T10:00:00Z",
            },
            {
                "pod_name": "pod-b",
                "queue_group": "queue_b",
                "status": "running",
                "run_id": "run_live",
                "current_step": "exp_b",
                "updated_at": "2026-03-19T12:05:00Z",
            },
        ],
    }

    active = effective_active_leases(
        registry,
        results_payload=results,
        leases_payload=leases,
        repo_root=tmp_path,
    )

    assert [lease["pod_name"] for lease in active] == ["pod-b"]
