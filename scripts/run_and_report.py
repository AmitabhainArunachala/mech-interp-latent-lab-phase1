#!/usr/bin/env python3
"""
Run a config-driven experiment, then validate + report results.

This script:
1) Runs preflight audit
2) Executes the experiment (src.pipelines.run)
3) Locates the new run_dir
4) Runs postrun validation
5) Verifies logging + posts to MCP monitor (optional)
6) Writes automation_report.json into the run_dir
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp_monitor.server import MCPServer


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_results_root(cfg: Dict[str, Any]) -> Path:
    results_root = (cfg.get("results") or {}).get("root") or "results"
    results_phase = (cfg.get("results") or {}).get("phase")
    if results_phase:
        return Path(results_root) / str(results_phase)
    return Path(results_root)


def _list_run_dirs(runs_root: Path) -> Set[Path]:
    if not runs_root.exists():
        return set()
    return {p for p in runs_root.iterdir() if p.is_dir()}


def _find_new_run_dir(before: Set[Path], after: Set[Path]) -> Optional[Path]:
    new_dirs = list(after - before)
    if not new_dirs:
        return None
    return max(new_dirs, key=lambda p: p.stat().st_mtime)


def _list_csv_files(run_dir: Path) -> List[str]:
    return sorted(str(p) for p in run_dir.rglob("*.csv"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_preflight(config_path: Path) -> bool:
    cmd = [
        sys.executable,
        "scripts/preflight_audit.py",
        "--config",
        str(config_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def _cuda_available() -> Optional[bool]:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return None


def _run_postrun_validation(results_path: Path) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/postrun_validator.py",
        "--results",
        str(results_path),
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        payload = json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError:
        payload = {"error": "postrun_validator_output_parse_failed", "stdout": result.stdout[-4000:]}
    payload["return_code"] = result.returncode
    if result.stderr:
        payload["stderr"] = result.stderr[-4000:]
    return payload


def run_and_report(
    config_path: Path,
    results_root_override: Optional[Path] = None,
    force: bool = False,
    mcp_enabled: bool = True,
) -> Dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    cfg = _load_json(config_path)

    results_root = results_root_override or _resolve_results_root(cfg)
    runs_root = results_root / "runs"

    preflight_passed = _run_preflight(config_path=config_path)
    if not preflight_passed and not force:
        return {
            "status": "aborted",
            "reason": "preflight_failed",
            "config": str(config_path),
        }

    params = cfg.get("params") or {}
    device = params.get("device", "auto")
    cuda_ok = _cuda_available()
    if device in ("cuda", "auto") and cuda_ok is False and not force:
        return {
            "status": "aborted",
            "reason": "cuda_unavailable",
            "config": str(config_path),
            "device": device,
        }

    before_dirs = _list_run_dirs(runs_root)

    cmd = [
        sys.executable,
        "-m",
        "src.pipelines.run",
        "--config",
        str(config_path),
    ]
    if results_root_override is not None:
        cmd.extend(["--results_root", str(results_root_override)])

    run_started_at = datetime.now().isoformat()

    # MCP: mark experiment started
    if mcp_enabled:
        try:
            MCPServer().start_experiment(
                experiment=cfg.get("experiment", "unknown"),
                model=(cfg.get("params") or {}).get("model", "unknown"),
                config_path=str(config_path),
            )
        except Exception:
            pass
    result = subprocess.run(cmd, capture_output=True, text=True)

    after_dirs = _list_run_dirs(runs_root)
    run_dir = _find_new_run_dir(before_dirs, after_dirs)

    if run_dir is None:
        return {
            "status": "failed",
            "reason": "run_dir_not_found",
            "config": str(config_path),
            "return_code": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }

    # Persist stdout/stderr for traceability
    _write_text(run_dir / "automation_stdout.txt", result.stdout)
    _write_text(run_dir / "automation_stderr.txt", result.stderr)

    # Post-run validation
    validation_dict = _run_postrun_validation(run_dir)

    # Verify logging (MCP server)
    mcp_server = MCPServer()
    logging_check = mcp_server.verify_logging(str(run_dir))

    # Load summary.json if present
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path) if summary_path.exists() else {}

    csv_files = _list_csv_files(run_dir)

    report = {
        "status": "completed" if result.returncode == 0 else "failed",
        "config": str(config_path),
        "run_dir": str(run_dir),
        "run_started_at": run_started_at,
        "run_finished_at": datetime.now().isoformat(),
        "return_code": result.returncode,
        "preflight_passed": preflight_passed,
        "postrun_validation": validation_dict,
        "logging_check": logging_check,
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "csv_files": csv_files,
    }

    # Write automation report into run dir
    report_path = run_dir / "automation_report.json"
    _write_text(report_path, json.dumps(report, indent=2))

    # MCP reporting
    if mcp_enabled:
        # Mark experiment start/end in MCP status (best effort)
        try:
            mcp_server.end_experiment(
                results_path=str(run_dir),
                success=result.returncode == 0,
                summary=summary if summary else None,
            )
        except Exception:
            pass

        # Post a concise finding
        verdict = validation_dict.get("verdict", "unknown")
        finding_text = (
            f"Run complete: {summary.get('experiment', cfg.get('experiment', 'unknown'))} "
            f"({summary.get('model', (cfg.get('params') or {}).get('model', 'unknown'))}). "
            f"Verdict: {verdict}. "
            f"CSV files: {len(csv_files)}. "
            f"Run dir: {run_dir}"
        )
        try:
            mcp_server.post_finding(
                source="cursor",
                finding_type="result",
                content=finding_text,
                evidence=str(report_path),
                priority="high" if verdict != "validated" else "medium",
            )
        except Exception:
            pass

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment and auto-report results")
    parser.add_argument("--config", type=Path, required=True, help="Path to config JSON")
    parser.add_argument("--results_root", type=Path, help="Override results root")
    parser.add_argument("--force", action="store_true", help="Run even if preflight fails")
    parser.add_argument("--no-mcp", action="store_true", help="Disable MCP reporting")
    args = parser.parse_args()

    report = run_and_report(
        config_path=args.config,
        results_root_override=args.results_root,
        force=args.force,
        mcp_enabled=not args.no_mcp,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
