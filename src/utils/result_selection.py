from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_best_persistent_patching_v3_dual(results_dir: Path) -> tuple[Path, dict[str, Any]] | tuple[None, None]:
    """
    Pick the most complete dual-layer artifact, breaking ties in favor of newer files.

    The directory can contain smoke, medium, and full runs. Lexicographic "first file"
    selection is therefore wrong once newer validation artifacts coexist with the older
    canonical run.
    """
    best_path: Path | None = None
    best_data: dict[str, Any] | None = None
    best_key: tuple[int, int, int, int, float, str] | None = None

    for path in sorted(results_dir.glob("persistent_patching_v3_dual_*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        aggregated = data.get("aggregated", {})
        total_turns = 0
        if isinstance(aggregated, dict):
            total_turns = max(
                (
                    condition.get("total_turns", 0)
                    for condition in aggregated.values()
                    if isinstance(condition, dict)
                ),
                default=0,
            )

        comparison_keys = data.get("comparisons", {})
        comparison_count = sum(
            1 for key in ("break_test", "induce_test", "sanity", "rv_session_contrasts")
            if key in comparison_keys
        )

        key = (
            int(data.get("n_sessions_per_condition", 0)),
            int(total_turns),
            int(data.get("max_turns_per_session", 0)),
            comparison_count,
            path.stat().st_mtime,
            path.name,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_path = path
            best_data = data

    if best_path is None or best_data is None:
        return None, None
    return best_path, best_data
