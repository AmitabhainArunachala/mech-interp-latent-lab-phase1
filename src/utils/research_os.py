from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / "configs" / "experiment_registry"
POD_LEASES_PATH = STATE_DIR / "pod_leases.json"
RESULTS_INDEX_PATH = STATE_DIR / "results_index.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def update_pod_lease(
    *,
    pod_name: str,
    host: str,
    port: int,
    session_name: str,
    queue_group: str,
    run_id: str,
    status: str,
    current_step: str = "",
    out_dir: str = "",
    notes: str = "",
) -> None:
    payload = _load_json(POD_LEASES_PATH, {"updated_at": utc_now_iso(), "leases": []})
    leases = payload.setdefault("leases", [])
    now = utc_now_iso()

    new_lease = {
        "pod_name": pod_name,
        "host": host,
        "port": int(port),
        "session_name": session_name,
        "queue_group": queue_group,
        "run_id": run_id,
        "status": status,
        "current_step": current_step,
        "out_dir": out_dir,
        "updated_at": now,
        "notes": notes,
    }

    replaced = False
    for idx, lease in enumerate(leases):
        if lease.get("pod_name") == pod_name:
            leases[idx] = new_lease
            replaced = True
            break
    if not replaced:
        leases.append(new_lease)

    payload["updated_at"] = now
    _write_json(POD_LEASES_PATH, payload)


def upsert_result(
    *,
    run_id: str,
    queue_group: str,
    experiment_id: str,
    status: str,
    artifact_path: str,
    model_family: str = "",
    model_name: str = "",
    config_path: str = "",
    prompt_contract: str = "",
    metric_path: str = "",
    claim_ids: list[str] | None = None,
) -> None:
    payload = _load_json(RESULTS_INDEX_PATH, {"updated_at": utc_now_iso(), "results": []})
    results = payload.setdefault("results", [])
    now = utc_now_iso()
    claim_ids = claim_ids or []

    entry = {
        "run_id": run_id,
        "queue_group": queue_group,
        "experiment_id": experiment_id,
        "status": status,
        "artifact_path": artifact_path,
        "model_family": model_family,
        "model_name": model_name,
        "config_path": config_path,
        "prompt_contract": prompt_contract,
        "metric_path": metric_path,
        "claim_ids": claim_ids,
        "updated_at": now,
    }

    replaced = False
    for idx, result in enumerate(results):
        if (
            result.get("run_id") == run_id
            and result.get("experiment_id") == experiment_id
        ):
            results[idx] = entry
            replaced = True
            break
    if not replaced:
        results.append(entry)

    payload["updated_at"] = now
    _write_json(RESULTS_INDEX_PATH, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AMIROS shared-state utility")
    sub = parser.add_subparsers(dest="command", required=True)

    lease = sub.add_parser("lease-update", help="Create or update a pod lease")
    lease.add_argument("--pod-name", required=True)
    lease.add_argument("--host", default="")
    lease.add_argument("--port", type=int, default=22)
    lease.add_argument("--session-name", required=True)
    lease.add_argument("--queue-group", required=True)
    lease.add_argument("--run-id", required=True)
    lease.add_argument("--status", required=True)
    lease.add_argument("--current-step", default="")
    lease.add_argument("--out-dir", default="")
    lease.add_argument("--notes", default="")

    result = sub.add_parser("result-upsert", help="Upsert a completed or running result entry")
    result.add_argument("--run-id", required=True)
    result.add_argument("--queue-group", required=True)
    result.add_argument("--experiment-id", required=True)
    result.add_argument("--status", required=True)
    result.add_argument("--artifact-path", required=True)
    result.add_argument("--model-family", default="")
    result.add_argument("--model-name", default="")
    result.add_argument("--config-path", default="")
    result.add_argument("--prompt-contract", default="")
    result.add_argument("--metric-path", default="")
    result.add_argument("--claim-id", action="append", default=[])

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "lease-update":
        update_pod_lease(
            pod_name=args.pod_name,
            host=args.host,
            port=args.port,
            session_name=args.session_name,
            queue_group=args.queue_group,
            run_id=args.run_id,
            status=args.status,
            current_step=args.current_step,
            out_dir=args.out_dir,
            notes=args.notes,
        )
        return 0

    if args.command == "result-upsert":
        upsert_result(
            run_id=args.run_id,
            queue_group=args.queue_group,
            experiment_id=args.experiment_id,
            status=args.status,
            artifact_path=args.artifact_path,
            model_family=args.model_family,
            model_name=args.model_name,
            config_path=args.config_path,
            prompt_contract=args.prompt_contract,
            metric_path=args.metric_path,
            claim_ids=args.claim_id,
        )
        return 0

    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
