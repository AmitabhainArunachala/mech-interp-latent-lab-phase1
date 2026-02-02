# Phase 0 Methodology: What we validate and how

Phase 0 is **not** “run more prompts.” It is targeted validation of:

1) **Measurement target**: is \(R_V\) tracking value-projection geometry specifically, or some correlated proxy?
2) **Confounds**: can syntax, topic, length, or window choice mimic the effect?
3) **Stability**: do the findings persist across small perturbations (prompt style) and parameter choices (window)?

---

## Canonical Phase 0 pipelines (reproducible)

### Pipeline A — Minimal Pairs (Semantics vs Syntax)
**Experiment:** `phase0_minimal_pairs`  
**Config:** `configs/phase0_minimal_pairs.json`

Measures whether \(R_V\) is stable under **syntax/style rephrasings** of the “same” intended meaning, and includes the **champion ablation set** (remove math / remove self-ref / remove fixed-point, etc.).

### Pipeline B — Metric Targets (What does R_V measure?)
**Experiment:** `phase0_metric_targets`  
**Config:** `configs/phase0_metric_targets.json`

Computes PR / \(R_V\) for multiple targets:
- PR of `v_proj.weight` (weight-space geometry)
- PR of `v_proj` outputs (current canonical measurement)
- PR of layer hidden outputs (residual-stream proxy)

Then reports correlations and group-level summaries across prompt families.

---

## “Existing Phase‑0‑adjacent work” (legacy scripts)

These scripts exist and are valuable, but their outputs were not consistently written into the phase0 folder:
- `reproduce_nov16_window_sweep.py` (window-size robustness)
- `experiment_grammar_confound.py` (question vs statement confound)
- `phase0_cross_baseline_control.py` (cross-baseline KV “gatekeeper” control)

Phase 0 going forward should use the canonical runner + configs so artifacts land in:
`results/phase0_metric_validation/runs/...`


