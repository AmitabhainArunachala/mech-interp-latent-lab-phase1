# Phase 0 Code (Canonical)

Phase 0 code is intentionally **not** stored as ad-hoc scripts in this folder.

The canonical code lives in the “living codebase”:
- `src/pipelines/phase0_minimal_pairs.py`
- `src/pipelines/phase0_metric_targets.py`

Run via:
- `src/pipelines/run.py` + `configs/phase0_*.json`

This guarantees:
- timestamped run folders under `results/phase0_metric_validation/runs/...`
- `config.json` snapshot, `summary.json`, `report.md`, and CSV artifacts.


