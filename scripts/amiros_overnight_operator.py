#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.mistral_program import utc_now_iso  # noqa: E402


PLAN_PATH = PROJECT_ROOT / "configs" / "experiment_registry" / "amiros_overnight_autonomy_plan.json"
STATE_PATH = PROJECT_ROOT / "configs" / "experiment_registry" / "amiros_overnight_autonomy_state.json"
LOG_PATH = PROJECT_ROOT / "docs" / "status" / "AMIROS_OVERNIGHT_AUTONOMY_LOG.md"
STATUS_PATH = PROJECT_ROOT / "docs" / "status" / "AMIROS_OVERNIGHT_AUTONOMY_STATUS.md"


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_log(line: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def write_status(lines: list[str]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_local(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def ssh_opts(pod: dict[str, Any]) -> list[str]:
    return [
        "-p",
        str(pod["port"]),
        "-i",
        str(Path(str(pod["ssh_key"])).expanduser()),
        "-o",
        "StrictHostKeyChecking=no",
    ]


def run_ssh(pod: dict[str, Any], remote_cmd: str) -> subprocess.CompletedProcess[str]:
    cmd = ["ssh", *ssh_opts(pod), str(pod["host"]), remote_cmd]
    return run_local(cmd)


def harvest_pod(pod: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["RUNPOD_HOST"] = str(pod["host"])
    env["RUNPOD_PORT"] = str(pod["port"])
    env["SSH_KEY"] = str(Path(str(pod["ssh_key"])).expanduser())
    return run_local(["bash", "scripts/harvest_runpod_research_os.sh"], env=env)


def sync_paths_to_pod(pod: dict[str, Any], paths: list[str]) -> list[str]:
    remote_repo = str(pod["remote_repo"])
    synced: list[str] = []
    for rel in paths:
        local_path = PROJECT_ROOT / rel
        remote_dir = f"{remote_repo}/{Path(rel).parent.as_posix()}/"
        cmd = [
            "rsync",
            "-az",
            "--no-owner",
            "--no-group",
            "-e",
            f"ssh {' '.join(shlex.quote(part) for part in ssh_opts(pod))}",
            str(local_path),
            f"{pod['host']}:{remote_dir}",
        ]
        proc = run_local(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"rsync failed for {rel}: {proc.stderr.strip()}")
        synced.append(rel)
    return synced


def launch_remote_step(pod: dict[str, Any], step: dict[str, Any], env_vars: dict[str, str]) -> subprocess.CompletedProcess[str]:
    remote_repo = str(pod["remote_repo"])
    exports = " ".join(f"export {k}={shlex.quote(v)};" for k, v in sorted(env_vars.items()))
    remote_cmd = (
        f"cd {shlex.quote(remote_repo)} && "
        f"{exports} "
        f"tmux new-session -d -s {shlex.quote(step['session_name'])} "
        f"{shlex.quote(f'bash {step['launcher']}')}"
    )
    return run_ssh(pod, remote_cmd)


def latest_path_for_glob(pattern: str) -> Path | None:
    matches = sorted(PROJECT_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def is_step_complete(step: dict[str, Any]) -> bool:
    pattern = str(step.get("artifact_glob") or "")
    return latest_path_for_glob(pattern) is not None if pattern else False


def current_session_running(pod: dict[str, Any], session_name: str) -> bool:
    proc = run_ssh(pod, f"tmux has-session -t {shlex.quote(session_name)}")
    return proc.returncode == 0


def top_conditions_from_induced_summary(summary_path: Path) -> list[str]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for condition, stats in summary.get("by_source_condition", {}).items():
        rows.append((condition, float(stats.get("bt_art_rate", 0.0)), -float(stats.get("mean_rv", 1e9))))
    rows.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return [row[0] for row in rows[:2]]


def should_launch_l5_subspace(summary_path: Path) -> bool:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    effects = summary.get("effects_vs_control", {})
    parallel_best = max(
        (float(v.get("recursive_bt_art_delta", -1e9)) for k, v in effects.items() if k.startswith("subspace3_parallel::")),
        default=-1e9,
    )
    orth_best = max(
        (float(v.get("recursive_bt_art_delta", -1e9)) for k, v in effects.items() if k.startswith("orthogonal_residual::")),
        default=-1e9,
    )
    return parallel_best >= orth_best - 0.01


def next_step_spec(lane: dict[str, Any], state_entry: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, str] | None, str]:
    steps = lane["steps"]
    current_idx = int(state_entry.get("current_step_index", 0))
    current_step = steps[current_idx]

    if current_idx + 1 >= len(steps):
      return None, None, "lane_complete"

    next_step = steps[current_idx + 1]
    env_vars = {k: str(v) for k, v in (next_step.get("env") or {}).items()}

    rule = str(next_step.get("decision_rule") or "")
    if rule == "induced_persistence_long":
        summary_path = latest_path_for_glob(str(current_step["artifact_glob"]))
        if summary_path is None:
            return None, None, "missing_induced_summary"
        top_conditions = top_conditions_from_induced_summary(summary_path)
        if not top_conditions:
            return None, None, "no_top_conditions"
        env_vars["SOURCE_CONDITIONS"] = ",".join(top_conditions)
        env_vars["NOTES"] = f"Longer persistence follow-up focused on top conditions from {summary_path.parent.name}"
    elif rule == "subspace_component_progression":
        prev_step = steps[current_idx]
        summary_path = latest_path_for_glob(str(prev_step["artifact_glob"]))
        if summary_path is None:
            return None, None, "missing_subspace_summary"
        if not should_launch_l5_subspace(summary_path):
            return None, None, "subspace_rule_stop"
        env_vars["NOTES"] = f"Early-layer follow-up after positive late-layer subspace signal in {summary_path.parent.name}"

    env_vars.setdefault("AMIROS_SESSION", str(next_step["session_name"]))
    env_vars.setdefault("AMIROS_POD_NAME", "")
    env_vars.setdefault("AMIROS_HOST", "")
    env_vars.setdefault("AMIROS_PORT", "")
    return next_step, env_vars, "launch_ready"


def initialize_state(plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    pods_state = state.setdefault("pods", {})
    for pod in plan.get("pods", []):
        pod_name = str(pod["pod_name"])
        if pod_name in pods_state:
            continue
        lane = next(l for l in plan["lanes"] if l["lane_id"] == pod["lane_id"])
        pods_state[pod_name] = {
            "lane_id": pod["lane_id"],
            "current_step_index": 0,
            "current_step_id": lane["steps"][0]["step_id"],
            "launched_steps": [],
            "completed_steps": [],
            "last_action": "initialized",
            "last_action_at": utc_now_iso(),
        }
    return state


def loop_once(plan: dict[str, Any], state: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    state = initialize_state(plan, state)
    status_lines = [
        "# AMIROS Overnight Autonomy Status",
        "",
        f"Generated: {utc_now_iso()}",
        "",
    ]

    lanes = {lane["lane_id"]: lane for lane in plan["lanes"]}

    for pod in plan["pods"]:
        pod_name = str(pod["pod_name"])
        lane = lanes[str(pod["lane_id"])]
        entry = state["pods"][pod_name]
        current_step = lane["steps"][entry["current_step_index"]]
        session_name = str(current_step["session_name"])
        running = current_session_running(pod, session_name)
        status_lines.append(f"## {pod_name}")
        status_lines.append(f"- lane: `{lane['lane_id']}`")
        status_lines.append(f"- current_step: `{current_step['step_id']}`")
        status_lines.append(f"- remote_session_running: `{running}`")

        if running:
            entry["last_action"] = "observed_running"
            entry["last_action_at"] = utc_now_iso()
            status_lines.append("- action: waiting for current step to finish")
            status_lines.append("")
            continue

        harvest = harvest_pod(pod)
        append_log(f"- {utc_now_iso()} harvest `{pod_name}` rc={harvest.returncode}")
        run_local(["python3", "scripts/nightly_summary.py"])

        if is_step_complete(current_step):
            if current_step["step_id"] not in entry["completed_steps"]:
                entry["completed_steps"].append(current_step["step_id"])
            next_step, env_vars, reason = next_step_spec(lane, entry)
            if next_step is None:
                entry["last_action"] = reason
                entry["last_action_at"] = utc_now_iso()
                status_lines.append(f"- action: `{reason}`")
                status_lines.append("")
                continue

            env_vars["AMIROS_POD_NAME"] = pod_name
            env_vars["AMIROS_HOST"] = str(pod["host"])
            env_vars["AMIROS_PORT"] = str(pod["port"])

            if dry_run:
                entry["last_action"] = f"would_launch:{next_step['step_id']}"
                entry["last_action_at"] = utc_now_iso()
                status_lines.append(f"- action: would launch `{next_step['step_id']}`")
                status_lines.append("")
                continue

            synced = sync_paths_to_pod(pod, list(next_step.get("sync_paths") or []))
            launched = launch_remote_step(pod, next_step, env_vars)
            if launched.returncode == 0:
                entry["current_step_index"] += 1
                entry["current_step_id"] = next_step["step_id"]
                entry["launched_steps"].append(next_step["step_id"])
                entry["last_action"] = f"launched:{next_step['step_id']}"
                entry["last_action_at"] = utc_now_iso()
                state["history"].append(
                    {
                        "at": utc_now_iso(),
                        "pod_name": pod_name,
                        "step_id": next_step["step_id"],
                        "synced": synced,
                        "env": env_vars,
                    }
                )
                append_log(f"- {utc_now_iso()} launched `{next_step['step_id']}` on `{pod_name}`")
                status_lines.append(f"- action: launched `{next_step['step_id']}`")
            else:
                entry["last_action"] = f"launch_failed:{next_step['step_id']}"
                entry["last_action_at"] = utc_now_iso()
                append_log(f"- {utc_now_iso()} launch FAILED `{next_step['step_id']}` on `{pod_name}`")
                status_lines.append(f"- action: launch failed `{next_step['step_id']}`")
            status_lines.append("")
            continue

        entry["last_action"] = "idle_no_artifact"
        entry["last_action_at"] = utc_now_iso()
        status_lines.append("- action: idle but current artifact missing; no launch")
        status_lines.append("")

    state["updated_at"] = utc_now_iso()
    write_status(status_lines)
    write_json(STATE_PATH, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Conservative overnight autopilot for AMIROS Mistral lanes")
    parser.add_argument("--plan", default=str(PLAN_PATH))
    parser.add_argument("--state", default=str(STATE_PATH))
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = load_json(Path(args.plan), {})
    state = load_json(Path(args.state), {"schema_version": "amiros_overnight_autonomy_state_v1", "updated_at": "", "pods": {}, "history": []})
    poll_seconds = int(args.poll_seconds or plan.get("poll_seconds") or 300)
    max_hours = float(args.max_hours or plan.get("max_hours_default") or 8)
    deadline = datetime.now(timezone.utc) + timedelta(hours=max_hours)

    append_log(f"# AMIROS overnight autonomy started {utc_now_iso()}")
    while True:
      state = loop_once(plan, state, dry_run=args.dry_run)
      if datetime.now(timezone.utc) >= deadline:
          append_log(f"- {utc_now_iso()} stop: max-hours reached")
          break
      time.sleep(poll_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
