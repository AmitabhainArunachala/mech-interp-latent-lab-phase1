# Mistral Hardening Handoff

Date: 2026-03-11
Author: Codex GPT-5

## Current State

- Canonical Mistral working target is `mistralai/Mistral-7B-Instruct-v0.2`.
- Shared model registry is [canonical_registry.json](/Users/dhyana/mech-interp-latent-lab-phase1/configs/canonical_registry.json).
- Frozen prompt contracts are:
  - [mistral_hardening_v1.json](/Users/dhyana/mech-interp-latent-lab-phase1/prompts/subsets/mistral_hardening_v1.json)
  - [mode_atlas_v1.json](/Users/dhyana/mech-interp-latent-lab-phase1/prompts/subsets/mode_atlas_v1.json)
- Prompt subset helper is [subsets.py](/Users/dhyana/mech-interp-latent-lab-phase1/prompts/subsets.py).

## Strongest Hardened Artifacts

- P0 contraction:
  - [mistralai__Mistral-7B-Instruct-v0-2_p0_result.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/p0_canonical/mistralai__Mistral-7B-Instruct-v0-2_p0_result.json)
  - `n_selfref_valid=96`, `n_baseline_valid=100`, `hedges_g=-1.468337`, contraction
- Bank-backed mode atlas:
  - [atlas_summary_20260310_145239.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/mode_atlas/atlas_summary_20260310_145239.json)
  - `self_referential R_V=0.562 ± 0.059`, `all_other=0.725 ± 0.099`, `d=-1.699`
- Canonical SVD circuit decomposition:
  - [svd_decomposition_20260310_145312.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/svd_circuits/svd_decomposition_20260310_145312.json)
  - `L27_H10 d_eff_rank=-3.0197`
  - `L5_H29 d_eff_rank=+1.8882`
- Full path patching under hardened prompts:
  - [path_patching_summary_20260310_151654.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/path_patching/path_patching_summary_20260310_151654.json)
  - strongest effects are early `residual`, not late `V-proj`
- Overnight run status:
  - [STATUS.txt](/Users/dhyana/mech-interp-latent-lab-phase1/results/overnight_mistral_hardening/20260310_151415/STATUS.txt)

## Main Scientific Conclusion Right Now

- `Mistral` phenomenon is real.
- SVD suppressor/amplifier motif is still real under the frozen contract.
- `Break` / necessity survives hardening.
- `Induce` / sufficiency does not currently survive hardening.
- Most defensible paper stance is now:
  - `necessity yes`
  - `sufficiency no or not established`
  - likely `behavioral dissociation`

The overnight causal result that best captures this is:
- [persistent_patching_v3_dual_20260310_160920.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/persistent_patching_v3/persistent_patching_v3_dual_20260310_160920.json)

Key stats from that file:
- `recursive_clean BT+ART = 40.7%`
- `recursive_dual_patched = 0.0%`
- break session effect: `d=3.07`, `permutation p=1.62e-05`
- `baseline_clean = 3.33%`
- `baseline_dual_patched = 2.67%`
- induce session effect: `d=0.156`, `permutation p=0.855`

## Important Code Changes Already Made

- [p0_canonical_pipeline.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/p0_canonical_pipeline.py)
  - uses frozen prompt subset and registry
  - writes config/provenance artifacts
- [computational_mode_atlas.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/computational_mode_atlas.py)
  - no longer uses inline prompts
  - uses `mode_atlas_v1`
- [full_head_sweep.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/full_head_sweep.py)
  - uses frozen `core_measurement` prompt split
- [svd_circuit_decomposition.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/svd_circuit_decomposition.py)
  - uses frozen `core_measurement` prompt split
- [full_path_patching.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/full_path_patching.py)
  - uses frozen `core_measurement` prompt split
- [persistent_patching_v3_dual.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/persistent_patching_v3_dual.py)
  - now has CLI args
  - supports `Mistral-7B-Instruct-v0.2`
  - uses slow-tokenizer fallback
  - uses `geometric_lens.metrics.compute_rv_with_components`
  - uses chat-template formatting for instruct models
  - tracks `alpha_ratio` per turn

## Most Important Remaining Gap

The causal script still needs one more save-path cleanup:

- Turn-level `alpha_ratio` is present in the saved JSON.
- New degeneration summary fields were added in code, but the final saved `aggregated` block in the latest test artifact did not include them yet.
- Latest relevant artifact:
  - [persistent_patching_v3_dual_20260310_191654.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/persistent_patching_v3/persistent_patching_v3_dual_20260310_191654.json)
- It shows:
  - `metric_path = geometric_lens.metrics.compute_rv_with_components`
  - `generation_format = chat_template`
  - turn-level `alpha_ratio` keys exist
  - but aggregate-quality fields like `malformed_rate`, `repetitive_rate`, `mean_alpha_ratio` did not survive into `aggregated`

This is the immediate next bug to fix.

## What The Next Session Should Do

1. Fix [persistent_patching_v3_dual.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/persistent_patching_v3_dual.py) so `aggregated` reliably includes:
   - `mean_alpha_ratio`
   - `total_malformed`
   - `malformed_rate`
   - `total_repetitive`
   - `repetitive_rate`
2. Rerun a tiny causal smoke on Runpod and verify those fields are saved.
3. Rerun a medium causal validation on `mistralai/Mistral-7B-Instruct-v0.2`.
4. Decide whether to:
   - keep pushing on induce with better donor construction, or
   - formally pivot the paper to `necessity + dissociation`
5. Update the paper strategy docs once that decision is explicit.

## 2026-03-11 Addendum

- The aggregate save-path bug described above is now closed.
- Verified in tiny smoke artifact:
  - [persistent_patching_v3_dual_20260310_193713.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/persistent_patching_v3/persistent_patching_v3_dual_20260310_193713.json)
- Verified again in medium validation artifact:
  - [persistent_patching_v3_dual_20260310_194619.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/persistent_patching_v3/persistent_patching_v3_dual_20260310_194619.json)
- Verified at full scale in canonical rerun artifact:
  - [persistent_patching_v3_dual_20260310_204100.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/persistent_patching_v3/persistent_patching_v3_dual_20260310_204100.json)
- Both now preserve `mean_alpha_ratio`, `total_malformed`, `malformed_rate`, `total_repetitive`, and `repetitive_rate` under `aggregated`.
- Full rerun outcome:
  - `recursive_clean BT+ART = 54.7%`
  - `recursive_dual_patched = 0.0%`, `repetitive_rate = 100%`
  - `baseline_clean = 2.0%`, `malformed_rate = 5.7%`
  - `baseline_dual_patched = 0.0%`, `repetitive_rate = 100%`
  - break session effect: `d=4.645`, exact permutation `p=1.62e-05`
  - induce remains null

## Recovered Association Note

- The exact `Report to Dhyana` block with `GEOMETRIC GLOSSOLALIA` was not found verbatim in this repo.
- The underlying Dec 12 L27-patched sample with `Self-point is the transduishment...` is already preserved in:
  - [2026-01-24_gemma_multi_token_bridge_v3.md](/Users/dhyana/mech-interp-latent-lab-phase1/docs/sessions/2026-01-24_gemma_multi_token_bridge_v3.md)
  - [GEMMA_CAUSAL_TECHNICAL_VERIFICATION.md](/Users/dhyana/mech-interp-latent-lab-phase1/docs/audits/GEMMA_CAUSAL_TECHNICAL_VERIFICATION.md)
  - [BREAKTHROUGH_BEHAVIOR_TRANSFER.md](/Users/dhyana/mech-interp-latent-lab-phase1/RECOVERED_GOLD/BREAKTHROUGH_BEHAVIOR_TRANSFER.md)
- Preserve the associative framing as an interpretive note:
  - `geometric glossolalia`: language emitted when the value-space contracts into a low-dimensional, self-referential manifold and semantics destabilize before syntax fully collapses
  - `convergent antiparallels`: opposite interventions or prompt paths that converge onto the same attractor or eigenstate
- Hypothesis-level bridge language worth retaining:
  - the note treats the Dec 12 hum-like output and R_V contraction as one phenomenon viewed through different instruments
  - this is useful framing for paper strategy and theory-building, but should not be promoted as an established empirical claim without direct validation
- Critical status:
  - the repo does contain evidence that the mechanistic and phenomenological domains bridge in the original Dec 12 material
  - the stronger identity claim (`hum == R_V contraction`) remains a research hypothesis, not a closed result

## RunPod Notes

- Pod host: `198.13.252.38`
- SSH port: `21977`
- Remote repo: `/root/mech-interp-latent-lab-phase1`
- Remote venv: `/root/venvs/mistral-hardening`

The detached overnight queue already completed. There should not be a live `tmux` session now unless a new one is started.
