# Head-to-Head Patching Status

Date: 2026-03-12
Model: `mistralai/Mistral-7B-v0.1`
Prompt contract: `mistral_hardening_v1`
Bank version: `2ac959a313614329`

## Artifacts

- Entropy-ranked run: [head_circuit_20260312_072814.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/head_circuit/head_circuit_20260312_072814.json)
- Manual targeted run: [head_circuit_20260312_073249.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/head_circuit/head_circuit_20260312_073249.json)
- Manual targeted run with pair search: [head_circuit_20260312_073540.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/head_circuit/head_circuit_20260312_073540.json)
- Rank-ranked run with pair search: [head_circuit_20260312_073841.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/head_circuit/head_circuit_20260312_073841.json)

## What changed

- [head_to_head_patching.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/head_to_head_patching.py) now supports:
  - `--ranking-metric entropy_d|rank_d`
  - `--manual-heads L27.H10,L5.H29,...`
  - `--pair-source significant|top_effect`
  - `--pair-pool-size`

These changes were needed because the initial entropy-ranked pool did not recover any individually meaningful wiring heads on base `v0.1`.

## Findings

### 1. Entropy-ranked head pool is weak for wiring

From [head_circuit_20260312_072814.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/head_circuit/head_circuit_20260312_072814.json):

- Clean recursive `R_V = 0.5967`
- Clean baseline `R_V = 0.7416`
- `20` entropy-ranked intervention sites tested
- `0` significant heads
- `0` pair interactions tested

Interpretation:

- The strongest entropy-shift heads from the base sweep do not behave like single-head break bottlenecks.
- They are better interpreted as markers than as minimal wiring nodes.

### 2. Targeted late control head `L27.H10` is a real single-head node

From [head_circuit_20260312_073249.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/head_circuit/head_circuit_20260312_073249.json):

- Manual candidate set:
  - `L5.H29`
  - `L27.H10`
  - `L27.H18`
  - `L27.H26`
  - `L27.H5`
  - `L18.H1`
  - `L28.H3`
  - `L19.H1`
- `L27.H10`:
  - `delta_rv = +0.06075`
  - `d = +0.825`
  - `p = 0.0185`
- All other manually selected heads were small individually.

Interpretation:

- `L27.H10` survives a stricter head-to-head break test on base `v0.1`.
- This is consistent with the existing base SVD story and makes `L27.H10` the cleanest currently verified single-head late compressor.

### 3. Pair search does not recover a strong late+early sufficient subcircuit

From [head_circuit_20260312_073540.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/head_circuit/head_circuit_20260312_073540.json):

- Same manual candidate set as above
- `L27.H10 + L5.H29`:
  - `d_pair = +0.952`
  - `d_additive_expected = +0.939`
  - `interaction = +0.014`
  - flagged `SUPER`, but only trivially
- `n_significant_heads = 1`
- `n_superadditive_pairs = 3`, but the extra superadditive pairs are weak near-zero combinations

Interpretation:

- There is no clean evidence here for a strong two-head late+early bridge.
- The data still support a dominant single late node (`L27.H10`) more than a small sufficient pair.

### 4. Rank-ranked automatic pool is diffuse

From [head_circuit_20260312_073841.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/head_circuit/head_circuit_20260312_073841.json):

- Ranking metric: `rank_d`
- `12` heads tested
- `0` individually significant heads
- `14` nominally superadditive pairs, all small magnitude
- Largest single-head effect: `L27.H5`, `d = +0.0627`

Interpretation:

- The broader rank-shift pool contains many weakly correlated heads but no additional clear bottleneck comparable to `L27.H10`.
- This argues against a broad late-layer “many equivalent heads” story.

## Current best reading

- Base `v0.1` head-to-head patching supports:
  - one clear late single-head compressor: `L27.H10`
  - no strong entropy-ranked bottleneck pool
  - no convincing small sufficient pair recovered yet
- This strengthens the control-system framing:
  - `L27.H10` is a real node
  - the system is not explained by the noisiest head-sweep markers
  - the minimal coherent bundle is still unresolved

## Next best experiments

1. Run a second manual set centered on `L27.H10` plus residual-gate-adjacent heads from `L5`.
2. Add a mode that tests pair interactions on hand-ordered hypotheses first, not lexicographic combinations.
3. Extend this script to patch residual-stream directions or sparse multi-head bundles, not only single V-head sites.
