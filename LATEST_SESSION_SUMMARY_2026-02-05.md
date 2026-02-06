---
title: "Latest Session Summary"
date: "2026-02-05"
scope: "Geometry–Behavior Gap + KV/Patching Temperature Sweep"
---

# Latest Session Summary (2026-02-05)

## What Ran Today

1) **L27 activation patching only**  
Run: `results/phase1_mechanism/runs/20260205_124324_rv_l27_activation_patching_bridge/`  
Summary pointer: `results/causal_bridge_summary.json`

2) **L27 + KV patching (T=0.0, top_p=0.95)**  
Run: `results/phase1_mechanism/runs/20260205_141146_rv_l27_kv_patching_bridge/`  
Summary pointer: `results/kv_patching_summary.json`

3) **Temperature sweep: L27 + KV (T=0.7)**  
Run (top_p=0.95): `results/phase1_mechanism/runs/20260205_151617_rv_l27_kv_patching_bridge/`  
Run (top_p=0.90): `results/phase1_mechanism/runs/20260205_152751_rv_l27_kv_patching_bridge/`  
Summary pointers:  
- `results/temp_sweep_t07_p095_summary.json`  
- `results/temp_sweep_t07_p09_summary.json`

## Verified Findings (from summary JSON)

### Geometry (R_V) transfers consistently
- `rv_recursive_mean ≈ 0.494`
- `rv_patched_mean ≈ 0.526`
- `rv_baseline_mean ≈ 0.672`
- `rv_p_value ≈ 6.96e-08`
- `rv_cohens_d ≈ -1.90`

### Behavior does not transfer (yet)
- L27 patching only: `behavior_strict_p_value ≈ 0.33` (ns)
- L27 + KV (T=0.0): `behavior_strict_p_value ≈ 0.71` (ns)
- L27 + KV (T=0.7, top_p=0.95): `behavior_strict_p_value ≈ 0.42` (ns)
- L27 + KV (T=0.7, top_p=0.90): `behavior_strict_p_value ≈ 0.58` (ns)

**Conclusion:** The geometric attractor is robust and transferable, but the behavioral attractor is resistant to patching, even with KV cache and temperature.

## Artifacts Confirmed Saved Locally

Top-level copies:
- `results/causal_bridge_summary.json`
- `results/per_sample.csv`
- `results/kv_patching_summary.json`
- `results/kv_patching_per_sample.csv`
- `results/temp_sweep_t07_p095_summary.json`
- `results/temp_sweep_t07_p095_per_sample.csv`
- `results/temp_sweep_t07_p09_summary.json`
- `results/temp_sweep_t07_p09_per_sample.csv`

Canonical run dirs (full artifacts):
- `results/phase1_mechanism/runs/20260205_124324_rv_l27_activation_patching_bridge/`
- `results/phase1_mechanism/runs/20260205_141146_rv_l27_kv_patching_bridge/`
- `results/phase1_mechanism/runs/20260205_151617_rv_l27_kv_patching_bridge/`
- `results/phase1_mechanism/runs/20260205_152751_rv_l27_kv_patching_bridge/`

## Interpretation

The bridge gap persists:
- **Geometry** transfers reliably (large effect size, highly significant).
- **Behavior** does not shift under current interventions.

This implies geometry alone is insufficient for behavior. The surgical sweep’s success likely depended on additional constraints (head-specific targeting, steering vector direction, or stronger diversity controls).

## Next Steps (Highest ROI)

1) **Head-specific patching (H18 + H26 only)**  
Rationale: surgical sweep identified these heads as critical. Full-layer patching may dilute behavioral effect.

2) **Multi-layer cascade patching (L25–L28)**  
Rationale: behavioral attractor may require broader residual stream influence.

3) **Repetition control**  
Introduce repetition penalty or top‑k constraints to break the looping attractor.

4) **Stronger intervention sweep (alpha scaling)**  
If behavior is thresholded, increase patch strength or mix weights.

---

**Hand-off note:** The above is a complete snapshot of today’s runs and their outcomes. Pick up from the Next Steps list.
