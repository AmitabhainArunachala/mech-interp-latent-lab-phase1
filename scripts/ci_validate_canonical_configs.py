#!/usr/bin/env python3
"""CI check: validate canonical config contract and experiment names.

This is intentionally lightweight (static JSON checks) and does not require model deps.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def extract_registry_experiments(registry_path: Path) -> set[str]:
    text = registry_path.read_text(encoding="utf-8")
    # Extract keys from the final return dict block: "exp_name": run_fn,
    return set(re.findall(r'^\s*"([a-zA-Z0-9_]+)"\s*:\s*', text, flags=re.MULTILINE))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    canonical_dir = repo_root / "configs" / "canonical"
    registry_path = repo_root / "src" / "pipelines" / "registry.py"

    errors: list[str] = []
    warnings: list[str] = []

    if not canonical_dir.exists():
        print("[FAIL] missing configs/canonical directory")
        return 1
    if not registry_path.exists():
        print("[FAIL] missing src/pipelines/registry.py")
        return 1

    known_experiments = extract_registry_experiments(registry_path)
    if not known_experiments:
        print("[FAIL] could not extract experiment names from registry.py")
        return 1

    # Active canonical surface only:
    # - root canonical configs used by blessed runner docs
    # - current seed matrix package
    config_files = sorted(canonical_dir.glob("*.json"))
    seed_matrix_dir = canonical_dir / "seed_bridge_2026_02_20"
    if seed_matrix_dir.exists():
        config_files.extend(sorted(seed_matrix_dir.glob("*.json")))
    if not config_files:
        errors.append("no canonical config files found")

    run_names: set[str] = set()

    for path in config_files:
        rel = path.relative_to(repo_root)
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"{rel}: invalid JSON ({e})")
            continue

        if not isinstance(cfg, dict):
            errors.append(f"{rel}: top-level must be JSON object")
            continue

        exp = cfg.get("experiment")
        if not isinstance(exp, str) or not exp.strip():
            errors.append(f"{rel}: missing/invalid 'experiment'")
        elif exp not in known_experiments:
            errors.append(f"{rel}: unknown experiment '{exp}'")

        run_name = cfg.get("run_name")
        if not isinstance(run_name, str) or not run_name.strip():
            warnings.append(f"{rel}: missing/empty 'run_name'")
        else:
            if run_name in run_names:
                errors.append(f"{rel}: duplicate run_name '{run_name}'")
            run_names.add(run_name)

        params = cfg.get("params")
        if not isinstance(params, dict):
            errors.append(f"{rel}: missing/invalid 'params' object")

        results = cfg.get("results")
        if not isinstance(results, dict):
            errors.append(f"{rel}: missing/invalid 'results' object")
        else:
            root = results.get("root")
            phase = results.get("phase")
            if not isinstance(root, str) or not root.strip():
                errors.append(f"{rel}: results.root must be non-empty string")
            if not isinstance(phase, str) or not phase.strip():
                errors.append(f"{rel}: results.phase must be non-empty string")

        # Contract-sensitive defaults expected in this repo
        if isinstance(params, dict):
            if "seed" not in params and "seed" not in cfg:
                warnings.append(f"{rel}: no explicit seed found (cfg.seed or params.seed)")

    print(f"Checked canonical configs: {len(config_files)}")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for w in warnings[:20]:
            print(f"  [WARN] {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")

    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:50]:
            print(f"  [FAIL] {e}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        return 1

    print("[PASS] canonical config contract is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
