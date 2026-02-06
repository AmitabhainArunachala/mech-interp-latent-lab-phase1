# Contract Violations Index

This index tracks known contract violations. Do not use these outputs in claims.

## Active Violations

| File | Violation | Notes |
|---|---|---|
| `results/canonical/final_results.json` | Single-layer PR mislabeled as R_V | Must recompute PR_late / PR_early |
| `rv_toolkit/rv_toolkit/metrics.py` | `compute_rv` returns PR, not ratio | Fix or deprecate |

## Actions Required

1. Recompute `final_results.json` with R_V ratio.
2. Archive legacy PR-only outputs after recompute.
3. Fix `rv_toolkit` or rename API to `compute_pr`.
