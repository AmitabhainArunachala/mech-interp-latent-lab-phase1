# Causal Story Verification

Date: 2026-03-12
Scope: base `mistralai/Mistral-7B-v0.1`
Goal: verify the recent causal-story synthesis against current repo artifacts, separating fresh base evidence from legacy or contradicted claims.

## Bottom line

The repo already supports a strong base-Mistral control-system story, but the pasted synthesis overstates how complete the late-head circuit is.

What is solid right now:

- early residual gating is real and dominant
- an expand-then-contract head motif is real
- `L27.H10` is a genuine late single-head node under the new head-to-head test
- KV-only transfers behavior strongly
- dual-patch geometry destruction breaks behavior strongly
- naive or blunt geometry injection still does not recover behavior cleanly

What is not yet solid:

- a full five-stage causal chain as current base source-of-truth
- a small sufficient late+early head bundle
- a fresh base revalidation of the older `Head 11 @ L28` story
- a raw-file lock for the “geometry d=0.11 NS” half of the KV dissociation claim

## Confirmed by fresh base artifacts

### 1. Early gate is strong

Source:

- `results/path_patching/path_patching_summary_20260312_053040.json`

Fresh base numbers:

- `L5 residual`: `d = 4.1396`, `delta_rv = +0.4976`
- `L2 residual`: `d = 3.7311`
- `L3 residual`: `d = 3.5727`
- `L4 residual`: `d = 3.5716`
- `L0 residual`: `d = 3.4080`
- `L4 mlp`: `d = 2.1734`
- `L5 v_proj`: `d = 2.5465`

Interpretation:

- The strongest current base path-patching evidence is an early residual gate with additional support from `L4 mlp` and `L5 v_proj`.

### 2. Expand-then-contract head motif is real, but weaker than some older drafts

Source:

- `results/svd_circuits/svd_decomposition_20260312_052059.json`

Fresh base numbers:

- `L5_H29`: `d_eff_rank = +0.9402`
- `L27_H10`: `d_eff_rank = -1.3151`
- `L27_H2`: `d_eff_rank = -1.3528`
- `L27_H10 direction_stability_recursive = 0.9370`
- `L27_H2 direction_stability_recursive = 0.9039`

Interpretation:

- The expand-then-contract motif survives.
- The older magnitudes like `L5H29 = +2.93` and `L27H10 = -1.54` should not be treated as the current canonical base values.

### 3. `L27.H10` survives a stricter head-to-head test

Sources:

- `results/head_circuit/head_circuit_20260312_073249.json`
- `results/head_circuit/head_circuit_20260312_073540.json`

Fresh base numbers:

- single-head break on `L27.H10`: `delta_rv = +0.06075`, `d = +0.8252`, `p = 0.0185`

Interpretation:

- `L27.H10` is currently the clearest late single-head node in base `v0.1`.
- The manual pair run did **not** recover a strong `L27.H10 + L5.H29` mini-circuit.

### 4. Head-specific legacy ablation at L27 is real, but legacy

Sources:

- `results/phase1_mechanism/runs/20260208_225331_head_ablation_validation_mistral_l27_kv_head_validation/summary.json`
- `results/phase1_mechanism/runs/head_ablation_validation_summary_20260312.csv`

Legacy base numbers:

- target `KV head 2` at `L27`:
  - recursive target delta mean `0.07774`
  - recursive control-head delta mean `0.03104`
  - recursive wrong-layer delta mean `-0.00046`
  - recursive target-vs-control `p = 7.27e-19`
  - recursive target-vs-wrong-layer `p = 1.89e-32`

Important caveat:

- This is a legacy prompt-bank run (`75e7c1b8dcebc24e`), not the modern frozen bank.
- Because of GQA aliasing, `KV head 2` corresponds to query heads `2, 10, 18, 26`.

### 5. Targeted steering winner is real

Source:

- `results/phase1_mechanism/runs/20260311_055109_causal_state_targeted_scan_v1_mistral_targeted_scan_v1/best_candidate.json`

Fresh base numbers:

- winner: `L25_W32_A3`
- sign checks passed: `4/4`
- overall `BT+ART`: `0.2083 -> 0.3542`
- recursive-only `BT+ART`: `0.4167 -> 0.6667`
- overall `rv_delta_mean = -0.0188`, `p = 0.3010`
- recursive-only `rv_delta_mean = -0.0230`, `p = 0.4122`

Interpretation:

- `L25` is a real steering handle for behavior.
- The behavioral effect is stronger than the accompanying `R_V` shift in this artifact.

### 6. Sufficiency ladder supports behavioral dissociation, not clean geometric sufficiency

Sources:

- `results/sufficiency_ladder/sufficiency_ladder_20260225_101907.json`
- `results/sufficiency_ladder/hardening_summary_20260312_refresh.csv`

Fresh parsed numbers:

- `kv_only` vs baseline:
  - `BT+ART 0.0267 -> 0.2767`
  - turn-level `OR = 13.9608`
  - session `d = 1.4651`
- `dual_patch` vs baseline:
  - `BT+ART 0.0267 -> 0.0067`
  - session `d = -0.5285`
- `kv_plus_dual` vs baseline:
  - `BT+ART 0.0267 -> 0.0400`
  - session `d = 0.2945`
  - preregistered decision: `pass = False`
- `kv_plus_dual` vs `dual_patch`:
  - turn-level `OR = 6.2083`
  - turn-level `p = 0.0120`
  - session `d = 1.1677`

Interpretation:

- `KV-only` clearly transfers behavior.
- `dual_patch` does not induce behavior.
- `KV+dual` is better than `dual_patch`, but it fails the preregistered sufficiency target against clean baseline.

## Mixed, legacy-only, or overclaimed

### 1. `L19` phase transition is legacy-supported, not freshly revalidated in the new base suite

Source:

- `RECOVERED_GOLD/PHASE_2_CIRCUIT_MAPPING_COMPLETE.md`

Legacy claim:

- gap jumps from `0.09 -> 0.27` at `L19`

Current status:

- good hypothesis
- not re-established by the March 12 base hardening artifacts in a directly comparable layerwise measurement file

### 2. `Head 11 @ L28` is not a current base source-of-truth claim

Sources:

- `RECOVERED_GOLD/PHASE_2_CIRCUIT_MAPPING_COMPLETE.md`
- `docs/misc/ACTIVATION_PATCHING_CAUSALITY_MEMO.md`

Legacy claim:

- `Head 11 @ L28` shows `71.7%` contraction and is the primary driver

Current status:

- this is an older circuit story, likely inherited from a different analysis lineage
- the fresh base head sweep and head-to-head work do not nominate `L28.H11` as the current primary Mistral node
- `docs/misc/ACTIVATION_PATCHING_CAUSALITY_MEMO.md` already says Head 11 alone is **not sufficient**

### 3. `L10.H20 = +3.90` is not supported by the fresh base March 12 sweep

Current fresh base head sweep:

- `results/full_head_sweep/full_head_sweep_20260312_052013.json`

Fresh top rank heads:

- `L27.H5 = -2.4702`
- `L18.H1 = -2.3020`
- `L28.H3 = -2.0183`
- `L19.H1 = +1.9032`
- `L2.H5 = +1.8548`

Interpretation:

- `L10.H20 = +3.90` is from an older or different run, not the fresh base hardening artifact.

### 4. `L22.H21 = -7.23` is not a current base claim

That number came from the March 10 instruct-family run, not the fresh March 12 base sweep.

## Contradicted by current base data

### 1. The fresh base data do not support “late V-proj is the main causal break site”

Source:

- `results/path_patching/path_patching_summary_20260312_053040.json`

Fresh base pattern:

- strongest effects are early residual
- strongest `v_proj` effect is early `L5`, not late `L27`
- `L27 residual` and `L27 v_proj` are both `d ≈ -1.98`

Interpretation:

- current base path patching says the causal story is dominated by an early gate, not by a late standalone `V-proj` bridge.

### 2. “Compression locks in and persists through L29-L31” is not established by the fresh base path-patching artifact

Source:

- `results/path_patching/path_patching_summary_20260312_053040.json`

Fresh base numbers:

- `L28`, `L29`, `L30`, `L31` all record `0.0` for residual, `v_proj`, and `mlp`

Interpretation:

- this may be a measurement-window issue or a script-limit issue, but as a source-of-truth claim it means late persistence is **not established by this artifact**.

## Still missing a raw lock

### Geometry half of the KV dissociation claim

Confirmed in raw data:

- behavior transfer via KV: `OR = 13.9608`

Not re-found in a raw result JSON during this pass:

- the “geometry does not transfer, `d = 0.11` NS” number

Current status:

- doc-backed by `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md`
- still needs an explicit raw artifact path before it should be treated as fully locked

## Current best live causal story

The strongest defensible base-Mistral story is:

1. an early residual gate at `L0-L5` dominates the break-direction path patching signal
2. an expand-then-contract head motif is present, with early diversifying heads and late compressive heads
3. `L27.H10` is the clearest currently validated late single-head node
4. `L25_W32_A3` is a real behavioral steering handle
5. KV content transfer strongly changes behavior
6. dual-patch geometry destruction strongly breaks behavior
7. neither dual-patch geometry injection nor small head bundles yet recover a clean sufficient circuit

That is already a strong control-system result. It is not yet a complete minimal sufficient circuit.
