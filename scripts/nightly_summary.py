#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.mistral_program import (  # noqa: E402
    PROGRAM_REGISTRY_PATH,
    POD_LEASES_PATH,
    RESULTS_INDEX_PATH,
    lease_is_stale,
    load_pod_leases,
    load_program_registry,
    load_results_index,
    ready_queue_units,
    reconcile_program_registry,
    save_program_registry,
)


CLAIM_REGISTRY_PATH = PROJECT_ROOT / "docs" / "status" / "CLAIM_REGISTRY.md"


def fmt_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def count_claim_statuses(path: Path) -> Counter[str]:
    if not path.exists():
        return Counter()
    text = path.read_text(encoding="utf-8")
    return Counter(re.findall(r"\|\s*(LOCKED|PROVISIONAL|INVALIDATED)\s*\|", text))


def latest_results_sorted(results_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        results_payload.get("results", []),
        key=lambda item: str(item.get("updated_at") or ""),
        reverse=True,
    )


def build_summary(*, sync_registry: bool) -> str:
    results_payload = load_results_index(RESULTS_INDEX_PATH)
    leases_payload = load_pod_leases(POD_LEASES_PATH)
    registry_payload = reconcile_program_registry(
        load_program_registry(PROGRAM_REGISTRY_PATH),
        results_payload=results_payload,
        leases_payload=leases_payload,
    )
    if sync_registry:
        save_program_registry(registry_payload, PROGRAM_REGISTRY_PATH)

    experiments = registry_payload.get("experiments", [])
    queue_units = registry_payload.get("queue_units", [])
    ready_units = ready_queue_units(registry_payload)
    latest_results = latest_results_sorted(results_payload)
    claim_counts = count_claim_statuses(CLAIM_REGISTRY_PATH)

    experiment_status_counts = Counter(
        str(exp.get("status") or "unknown") for exp in experiments
    )
    queue_status_counts = Counter(
        str(unit.get("status") or "unknown") for unit in queue_units
    )

    active_leases = [
        lease for lease in leases_payload.get("leases", []) if lease.get("status") == "running"
    ]
    stale_leases = [lease for lease in active_leases if lease_is_stale(lease)]
    missing_artifacts = [
        exp
        for exp in experiments
        if exp.get("status") == "completed" and not exp.get("artifact_exists_local")
    ]
    blocked_units = [unit for unit in queue_units if unit.get("status") == "blocked"]
    orphan_results = [
        result
        for result in latest_results
        if not any(
            exp.get("experiment_id") == result.get("experiment_id")
            for exp in experiments
        )
    ]

    lines: list[str] = []
    lines.append("# Nightly Summary")
    lines.append("")
    lines.append(f"Generated: {fmt_ts()}")
    lines.append("")
    lines.append("## Program Status")
    lines.append(f"- Registry: `{PROGRAM_REGISTRY_PATH.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- Results index: `{RESULTS_INDEX_PATH.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- Pod leases: `{POD_LEASES_PATH.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- Queue units: `{len(queue_units)}` total, `{queue_status_counts.get('completed', 0)}` completed, `{queue_status_counts.get('running', 0)}` running, `{queue_status_counts.get('queued', 0)}` queued, `{queue_status_counts.get('blocked', 0)}` blocked, `{queue_status_counts.get('failed', 0)}` failed")
    lines.append(f"- Experiments: `{len(experiments)}` total, `{experiment_status_counts.get('completed', 0)}` completed, `{experiment_status_counts.get('running', 0)}` running, `{experiment_status_counts.get('queued', 0)}` queued, `{experiment_status_counts.get('failed', 0)}` failed")
    if claim_counts:
        lines.append(f"- Claim registry: `{claim_counts.get('LOCKED', 0)}` locked, `{claim_counts.get('PROVISIONAL', 0)}` provisional, `{claim_counts.get('INVALIDATED', 0)}` invalidated")
    else:
        lines.append("- Claim registry counts unavailable.")

    lines.append("")
    lines.append("## Active Pods")
    if active_leases:
        for lease in active_leases:
            stale_tag = " [STALE]" if lease in stale_leases else ""
            lines.append(
                f"- `{lease.get('pod_name')}`{stale_tag}: queue `{lease.get('queue_group')}`, run `{lease.get('run_id')}`, step `{lease.get('current_step')}`, updated `{lease.get('updated_at')}`"
            )
    else:
        lines.append("- No running pod leases recorded.")

    lines.append("")
    lines.append("## Ready Next Queue Units")
    if ready_units:
        for unit in ready_units[:6]:
            lines.append(
                f"- `{unit.get('queue_unit_id')}`: stage `{unit.get('stage')}`, queue `{unit.get('queue_group')}`, priority `{unit.get('priority')}`, expected `{unit.get('expected_runtime_hours')}`h, launcher `{(unit.get('launcher') or {}).get('path', '')}`"
            )
    else:
        lines.append("- No ready queue units. Either the queue is exhausted or dependencies are still blocked.")

    lines.append("")
    lines.append("## Latest Results")
    if latest_results:
        for result in latest_results[:10]:
            lines.append(
                f"- `{result.get('experiment_id')}` [{result.get('status')}] -> `{result.get('artifact_path')}`"
            )
    else:
        lines.append("- No indexed results yet.")

    lines.append("")
    lines.append("## State Warnings")
    if stale_leases:
        for lease in stale_leases:
            lines.append(
                f"- Stale running lease: `{lease.get('pod_name')}` queue `{lease.get('queue_group')}` last updated `{lease.get('updated_at')}`"
            )
    if missing_artifacts:
        for exp in missing_artifacts[:10]:
            lines.append(
                f"- Completed without local artifact: `{exp.get('experiment_id')}` expected `{exp.get('artifact_path') or exp.get('artifact_glob')}`"
            )
    if blocked_units:
        for unit in blocked_units[:10]:
            blockers = ", ".join(
                warning.replace("blocked_by:", "")
                for warning in unit.get("warnings", [])
                if str(warning).startswith("blocked_by:")
            ) or "unknown"
            lines.append(
                f"- Blocked queue unit: `{unit.get('queue_unit_id')}` waiting on `{blockers}`"
            )
    if orphan_results:
        for result in orphan_results[:10]:
            lines.append(
                f"- Result not represented in registry: `{result.get('experiment_id')}` -> `{result.get('artifact_path')}`"
            )
    if not any([stale_leases, missing_artifacts, blocked_units, orphan_results]):
        lines.append("- No registry/state mismatches detected.")

    lines.append("")
    lines.append("## Recommended Next Actions")
    if stale_leases:
        lines.append("- Reconcile or clear stale leases before trusting any queue status.")
    elif active_leases:
        lines.append("- Let the active queue finish before assigning new heavy work to the same pod.")
    elif ready_units:
        next_unit = ready_units[0]
        lines.append(
            f"- Next clean launch is `{next_unit.get('queue_unit_id')}` via `{(next_unit.get('launcher') or {}).get('path', '')}`."
        )
    else:
        lines.append("- No launch is ready; add the next approved queue unit or mark completed work in the registry.")
    lines.append("- Harvest remote artifacts before updating paper-facing claims.")
    lines.append("- Treat orphan or stale state as operational debt, not as evidence.")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Mistral nightly summary.")
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "docs" / "status" / "NIGHTLY_SUMMARY.md"),
        help="Output markdown path",
    )
    parser.add_argument(
        "--no-sync-registry",
        action="store_true",
        help="Do not write the reconciled registry back to disk",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        build_summary(sync_registry=not args.no_sync_registry),
        encoding="utf-8",
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
