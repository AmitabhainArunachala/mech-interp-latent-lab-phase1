#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.mistral_program import (  # noqa: E402
    PROGRAM_REGISTRY_PATH,
    experiment_map,
    latest_result_by_experiment,
    load_pod_leases,
    load_program_registry,
    load_results_index,
    ready_queue_units,
    reconcile_program_registry,
    save_program_registry,
    set_experiment_status,
    set_queue_unit_status,
    utc_now_iso,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Mistral overnight queue units from the program registry."
    )
    parser.add_argument(
        "--queue-group",
        default="",
        help="Restrict launches to one queue_group",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Restrict launches to one or more stages",
    )
    parser.add_argument(
        "--max-queue-units",
        type=int,
        default=3,
        help="Maximum number of ready queue units to launch in order",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Override output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ready queue order without launching anything",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue to the next queue unit if one fails or is partial",
    )
    return parser


def build_command(unit: dict[str, Any]) -> list[str]:
    launcher = unit.get("launcher") or {}
    launch_type = str(launcher.get("type") or "bash")
    args = [str(arg) for arg in launcher.get("args", []) or []]

    if launch_type == "bash":
        return ["bash", str(launcher["path"]), *args]
    if launch_type == "python_script":
        return [sys.executable, str(launcher["path"]), *args]
    if launch_type == "module":
        return [sys.executable, "-m", str(launcher["module"]), *args]

    raise ValueError(f"Unsupported launcher type: {launch_type}")


def run_with_tee(
    cmd: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        return process.wait()


def write_status_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def collect_outcomes(
    queue_unit: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    reconciled = reconcile_program_registry(
        load_program_registry(PROGRAM_REGISTRY_PATH),
        results_payload=load_results_index(),
        leases_payload=load_pod_leases(),
    )
    experiments = experiment_map(reconciled)
    outcomes: list[dict[str, Any]] = []
    statuses: list[str] = []

    for experiment_id in queue_unit.get("experiment_ids", []) or []:
        exp = experiments.get(str(experiment_id), {})
        status = str(exp.get("status") or "queued")
        statuses.append(status)
        outcomes.append(
            {
                "experiment_id": experiment_id,
                "status": status,
                "artifact_path": exp.get("artifact_path", ""),
                "artifact_exists_local": bool(exp.get("artifact_exists_local")),
                "warnings": list(exp.get("warnings", [])),
                "latest_result": exp.get("latest_result", {}),
            }
        )

    if outcomes and all(item["status"] == "completed" for item in outcomes):
        queue_status = "completed"
    elif any(item["status"] == "failed" for item in outcomes):
        queue_status = "failed"
    else:
        queue_status = "partial"

    return reconciled, outcomes, queue_status


def mark_unit_running(
    registry_payload: dict[str, Any],
    *,
    unit: dict[str, Any],
    queue_run_id: str,
) -> dict[str, Any]:
    payload = set_queue_unit_status(
        registry_payload,
        queue_unit_id=str(unit["queue_unit_id"]),
        status="running",
        run_id=queue_run_id,
        note="launcher_active",
    )
    for experiment_id in unit.get("experiment_ids", []) or []:
        payload = set_experiment_status(
            payload,
            experiment_id=str(experiment_id),
            status="running",
            run_id=queue_run_id,
        )
    return payload


def apply_outcomes(
    registry_payload: dict[str, Any],
    *,
    unit: dict[str, Any],
    outcomes: list[dict[str, Any]],
    queue_status: str,
    queue_run_id: str,
) -> dict[str, Any]:
    payload = registry_payload
    for outcome in outcomes:
        payload = set_experiment_status(
            payload,
            experiment_id=str(outcome["experiment_id"]),
            status=str(outcome["status"]),
            artifact_path=str(outcome.get("artifact_path") or ""),
            run_id=queue_run_id,
        )
    payload = set_queue_unit_status(
        payload,
        queue_unit_id=str(unit["queue_unit_id"]),
        status=queue_status,
        run_id=queue_run_id,
        note="artifacts_reconciled",
    )
    return payload


def main() -> int:
    args = build_parser().parse_args()
    queue_run_id = Path(utc_now_iso().replace("-", "").replace(":", "")).name
    queue_run_id = queue_run_id.replace("T", "_").replace("Z", "")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else PROJECT_ROOT / "results" / "mistral_program_queue" / queue_run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "STATUS.txt"
    manifest_path = out_dir / "manifest.json"

    write_status_line(status_path, f"run_id={queue_run_id}")
    write_status_line(status_path, f"registry_path={PROGRAM_REGISTRY_PATH}")
    write_status_line(status_path, f"started_utc={utc_now_iso()}")

    registry = reconcile_program_registry(load_program_registry(PROGRAM_REGISTRY_PATH))
    save_program_registry(registry)

    ready = ready_queue_units(
        registry,
        queue_group=(args.queue_group or None),
        allowed_stages=set(args.stage) if args.stage else None,
    )
    selected = ready[: max(args.max_queue_units, 0)]

    manifest: dict[str, Any] = {
        "run_id": queue_run_id,
        "started_utc": utc_now_iso(),
        "registry_path": str(PROGRAM_REGISTRY_PATH),
        "selected_queue_units": [unit.get("queue_unit_id") for unit in selected],
        "results": [],
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    if not selected:
        write_status_line(status_path, "no_ready_queue_units=1")
        manifest["finished_utc"] = utc_now_iso()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("No ready queue units.")
        return 0

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))

    for unit in selected:
        unit_id = str(unit["queue_unit_id"])
        write_status_line(status_path, f">>> START {unit_id} {utc_now_iso()}")
        registry = mark_unit_running(registry, unit=unit, queue_run_id=queue_run_id)
        save_program_registry(registry)

        cmd = build_command(unit)
        log_path = out_dir / f"{unit_id}.log"
        rc = run_with_tee(cmd, log_path=log_path, env=env)

        registry, outcomes, queue_status = collect_outcomes(unit)
        if rc != 0 and queue_status != "completed":
            queue_status = "failed"
        elif rc == 0 and queue_status == "partial":
            queue_status = "partial"

        registry = apply_outcomes(
            registry,
            unit=unit,
            outcomes=outcomes,
            queue_status=queue_status,
            queue_run_id=queue_run_id,
        )
        save_program_registry(registry)

        manifest["results"].append(
            {
                "queue_unit_id": unit_id,
                "command": cmd,
                "return_code": rc,
                "queue_status": queue_status,
                "outcomes": outcomes,
                "log_path": str(log_path.relative_to(PROJECT_ROOT)),
            }
        )
        write_status_line(
            status_path,
            f">>> {queue_status.upper()} {unit_id} rc={rc} {utc_now_iso()}",
        )

        if queue_status in {"failed", "partial"} and not args.continue_on_failure:
            manifest["stopped_on"] = unit_id
            break

    manifest["finished_utc"] = utc_now_iso()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
