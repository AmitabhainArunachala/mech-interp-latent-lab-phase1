from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / "configs" / "experiment_registry"
PROGRAM_REGISTRY_PATH = STATE_DIR / "mistral_program_registry.json"
POD_LEASES_PATH = STATE_DIR / "pod_leases.json"
RESULTS_INDEX_PATH = STATE_DIR / "results_index.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return deepcopy(default)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else deepcopy(default)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_program_registry(
    path: Path = PROGRAM_REGISTRY_PATH,
) -> dict[str, Any]:
    payload = load_json(path, {"experiments": [], "queue_units": []})
    payload.setdefault("experiments", [])
    payload.setdefault("queue_units", [])
    return payload


def save_program_registry(
    payload: dict[str, Any],
    path: Path = PROGRAM_REGISTRY_PATH,
) -> None:
    write_json(path, payload)


def load_results_index(
    path: Path = RESULTS_INDEX_PATH,
) -> dict[str, Any]:
    payload = load_json(path, {"results": [], "updated_at": ""})
    payload.setdefault("results", [])
    return payload


def load_pod_leases(
    path: Path = POD_LEASES_PATH,
) -> dict[str, Any]:
    payload = load_json(path, {"leases": [], "updated_at": ""})
    payload.setdefault("leases", [])
    return payload


def experiment_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(exp.get("experiment_id")): exp
        for exp in payload.get("experiments", [])
        if exp.get("experiment_id")
    }


def queue_unit_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(unit.get("queue_unit_id")): unit
        for unit in payload.get("queue_units", [])
        if unit.get("queue_unit_id")
    }


def latest_result_by_experiment(
    results_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for result in results_payload.get("results", []):
        experiment_id = str(result.get("experiment_id") or "")
        if not experiment_id:
            continue
        current = latest.get(experiment_id)
        stamp = str(result.get("updated_at") or "")
        if current is None or stamp >= str(current.get("updated_at") or ""):
            latest[experiment_id] = result
    return latest


def latest_lease_by_queue_group(
    leases_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for lease in leases_payload.get("leases", []):
        queue_group = str(lease.get("queue_group") or "")
        if not queue_group:
            continue
        current = latest.get(queue_group)
        stamp = str(lease.get("updated_at") or "")
        if current is None or stamp >= str(current.get("updated_at") or ""):
            latest[queue_group] = lease
    return latest


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_artifact_glob(
    artifact_glob: str,
    *,
    repo_root: Path = PROJECT_ROOT,
) -> Path | None:
    if not artifact_glob:
        return None
    matches = sorted(
        repo_root.glob(artifact_glob),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    return matches[-1] if matches else None


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def lease_is_stale(
    lease: dict[str, Any],
    *,
    now: datetime | None = None,
    threshold_minutes: int = 90,
) -> bool:
    now = now or datetime.now(timezone.utc)
    stamp = _parse_iso(str(lease.get("updated_at") or ""))
    if stamp is None:
        return True
    return now - stamp > timedelta(minutes=threshold_minutes)


def dependencies_satisfied(
    queue_unit: dict[str, Any],
    *,
    units: dict[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    for dep_id in queue_unit.get("depends_on_queue_units", []) or []:
        dep = units.get(str(dep_id))
        if dep is None:
            blockers.append(f"missing:{dep_id}")
            continue
        if dep.get("status") != "completed":
            blockers.append(str(dep_id))
    return (not blockers, blockers)


def reconcile_program_registry(
    registry_payload: dict[str, Any],
    *,
    results_payload: dict[str, Any] | None = None,
    leases_payload: dict[str, Any] | None = None,
    repo_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    payload = deepcopy(registry_payload)
    payload.setdefault("experiments", [])
    payload.setdefault("queue_units", [])
    payload["reconciled_at"] = utc_now_iso()

    results_payload = results_payload or load_results_index()
    leases_payload = leases_payload or load_pod_leases()
    latest_results = latest_result_by_experiment(results_payload)
    latest_leases = latest_lease_by_queue_group(leases_payload)
    now = datetime.now(timezone.utc)

    for exp in payload.get("experiments", []):
        experiment_id = str(exp.get("experiment_id") or "")
        result = latest_results.get(experiment_id)
        artifact_path = str(exp.get("artifact_path") or "")
        artifact_glob = str(exp.get("artifact_glob") or "")
        warnings: list[str] = []

        if result is not None:
            exp["latest_result"] = {
                "run_id": result.get("run_id", ""),
                "status": result.get("status", ""),
                "updated_at": result.get("updated_at", ""),
                "artifact_path": result.get("artifact_path", ""),
            }
            if result.get("status"):
                exp["status"] = result["status"]
            if result.get("artifact_path"):
                artifact_path = str(result["artifact_path"])
                exp["artifact_path"] = artifact_path

        resolved = None
        if artifact_path:
            candidate = repo_root / artifact_path
            if candidate.exists():
                resolved = candidate
        if resolved is None and artifact_glob:
            resolved = resolve_artifact_glob(artifact_glob, repo_root=repo_root)
            if resolved is not None:
                exp["artifact_path"] = relative_to_repo(resolved)
                artifact_path = exp["artifact_path"]

        exp["artifact_exists_local"] = bool(resolved and resolved.exists())
        if exp.get("status") == "completed" and not exp["artifact_exists_local"]:
            warnings.append("completed_without_local_artifact")

        queue_group = str(exp.get("queue_group") or "")
        lease = latest_leases.get(queue_group)
        if exp.get("status") == "running" and lease is None:
            warnings.append("running_without_lease")
        if lease is not None and lease.get("status") == "running":
            exp["active_lease"] = {
                "pod_name": lease.get("pod_name", ""),
                "run_id": lease.get("run_id", ""),
                "current_step": lease.get("current_step", ""),
                "updated_at": lease.get("updated_at", ""),
            }
            if lease_is_stale(lease, now=now):
                warnings.append("stale_lease")
        else:
            exp.pop("active_lease", None)
        exp["warnings"] = warnings

    experiments = experiment_map(payload)
    units = queue_unit_map(payload)

    for unit in payload.get("queue_units", []):
        queue_group = str(unit.get("queue_group") or "")
        unit_experiments = [
            experiments[exp_id]
            for exp_id in unit.get("experiment_ids", []) or []
            if exp_id in experiments
        ]
        lease = latest_leases.get(queue_group)
        statuses = {str(exp.get("status") or "queued") for exp in unit_experiments}

        deps_ok, blockers = dependencies_satisfied(unit, units=units)
        warnings: list[str] = []
        if lease is not None and lease.get("status") == "running":
            unit["status"] = "running"
            if lease_is_stale(lease, now=now):
                warnings.append("stale_lease")
        elif unit_experiments and all(
            exp.get("status") == "completed" for exp in unit_experiments
        ):
            unit["status"] = "completed"
        elif any(exp.get("status") == "failed" for exp in unit_experiments):
            unit["status"] = "failed"
        elif not deps_ok:
            unit["status"] = "blocked"
        elif any(status == "running" for status in statuses):
            unit["status"] = "running"
        else:
            unit["status"] = unit.get("status") or "queued"

        if blockers:
            warnings.extend(f"blocked_by:{item}" for item in blockers)
        if not unit_experiments:
            warnings.append("no_registered_experiments")
        if unit.get("status") == "completed":
            missing = [
                exp.get("experiment_id")
                for exp in unit_experiments
                if not exp.get("artifact_exists_local")
            ]
            if missing:
                warnings.append(
                    "missing_artifacts:" + ",".join(str(item) for item in missing)
                )

        if lease is not None:
            unit["active_lease"] = {
                "pod_name": lease.get("pod_name", ""),
                "run_id": lease.get("run_id", ""),
                "current_step": lease.get("current_step", ""),
                "updated_at": lease.get("updated_at", ""),
            }
        else:
            unit.pop("active_lease", None)

        unit["warnings"] = warnings

    return payload


def ready_queue_units(
    registry_payload: dict[str, Any],
    *,
    queue_group: str | None = None,
    allowed_stages: set[str] | None = None,
) -> list[dict[str, Any]]:
    payload = reconcile_program_registry(registry_payload)
    units = queue_unit_map(payload)
    ready: list[dict[str, Any]] = []

    for unit in payload.get("queue_units", []):
        if queue_group and unit.get("queue_group") != queue_group:
            continue
        if allowed_stages and unit.get("stage") not in allowed_stages:
            continue
        if unit.get("status") != "queued":
            continue
        deps_ok, blockers = dependencies_satisfied(unit, units=units)
        if not deps_ok:
            unit["dependency_blockers"] = blockers
            continue
        ready.append(unit)

    return sorted(
        ready,
        key=lambda item: (
            int(item.get("priority", 9999)),
            str(item.get("stage") or ""),
            str(item.get("queue_unit_id") or ""),
        ),
    )


def set_queue_unit_status(
    registry_payload: dict[str, Any],
    *,
    queue_unit_id: str,
    status: str,
    run_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    payload = deepcopy(registry_payload)
    for unit in payload.get("queue_units", []):
        if unit.get("queue_unit_id") != queue_unit_id:
            continue
        unit["status"] = status
        unit["last_status_at"] = utc_now_iso()
        if run_id is not None:
            unit["last_run_id"] = run_id
        if note is not None:
            unit["last_note"] = note
        break
    return payload


def set_experiment_status(
    registry_payload: dict[str, Any],
    *,
    experiment_id: str,
    status: str,
    artifact_path: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    payload = deepcopy(registry_payload)
    for exp in payload.get("experiments", []):
        if exp.get("experiment_id") != experiment_id:
            continue
        exp["status"] = status
        exp["last_status_at"] = utc_now_iso()
        if artifact_path is not None:
            exp["artifact_path"] = artifact_path
        if run_id is not None:
            exp["last_run_id"] = run_id
        break
    return payload
