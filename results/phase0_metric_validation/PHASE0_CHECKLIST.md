# Phase 0 Checklist (Done vs Missing) — with receipts

This is the “memory prosthetic.” If it’s not linked here, it doesn’t count as Phase 0.

---

## ✅ DONE (exists in repo)

### ✅ Window-size robustness (script exists)
- **Script:** `reproduce_nov16_window_sweep.py`
- **What it tests:** window sizes `[16, 32, 64]` on the same prompt pairs.
- **Status:** script exists; canonical Phase‑0 runs should emit CSV into `results/phase0_metric_validation/runs/...` via the runner.

### ✅ Grammar / question confound (script exists)
- **Script:** `experiment_grammar_confound.py`
- **What it tests:** do self-referential *statements* contract like questions?
- **Status:** script exists; outputs not currently tracked in `results/phase0_metric_validation/`.

### ✅ Cross-baseline “Gatekeeper” KV control (script exists)
- **Script:** `phase0_cross_baseline_control.py`
- **What it tests:** baseline→baseline KV mixing should not induce recursive behavior.
- **Status:** script exists; writes to a hardcoded path (`results/dec11_evening/...`) which is not present in repo.

### ✅ Champion calibration (numbers exist)
- **Docs:** `VALIDATION_REPORT.md`, `KITCHEN_SINK_REPORT.md`
- **Key number:** `hybrid_l5_math_01` achieves **R_V(L27)=0.5088** with std≈0 across 10 runs.

---

## ✅ DONE (canonical Phase 0 pipelines) — NEW STANDARD

### Pipeline A: Minimal Pairs (Semantics vs Syntax)
- **Experiment:** `phase0_minimal_pairs`
- **Config:** `configs/phase0_minimal_pairs.json`
- **Outputs:** `results/phase0_metric_validation/runs/.../phase0_minimal_pairs.csv`

### Pipeline B: Metric Targets (What does R_V measure?)
- **Experiment:** `phase0_metric_targets`
- **Config:** `configs/phase0_metric_targets.json`
- **Outputs:** `results/phase0_metric_validation/runs/.../phase0_metric_targets.csv`

---

## ❌ MISSING (required to “close” Phase 0)

### ❌ V-matrix / V-output / hidden comparison (executed + logged)
We need a run artifact that directly measures and correlates:
- PR(`v_proj.weight`)
- PR(`v_proj` output)
- PR(hidden states)

> This is implemented by `phase0_metric_targets` and must be executed per-model to count.

### ❌ Convex hull verification (executed + logged)
Extract α and V and verify convex combination properties and any “distance to hull boundary” hypothesis.

### ❌ Semantic-equivalent minimal pairs at scale (executed + logged)
Not just “grammar confound,” but paraphrase-style minimal pairs that keep meaning constant.


