# Mistral Base v0.1 Head-to-Head Status

Date: 2026-03-12
Model: `mistralai/Mistral-7B-v0.1`
Prompt contract: `mistral_hardening_v1` / `core_measurement`

## Canonical artifact

- Recommended current artifact: `results/head_circuit/head_circuit_20260312_075244.json`
- Launcher run: `results/head_circuit_runs/20260312_075024/`
- Head sweep source: `results/full_head_sweep/full_head_sweep_20260302_074757.json`

## What was run

1. Ranked exploratory pass
   - Artifact: `results/head_circuit/head_circuit_20260312_074253.json`
   - Config: `n_prompts=30`, `top_k=24`, `pair_source=top_effect`, `pair_pool_size=8`, `pair_prompt_limit=20`
   - Outcome: surfaced `L05.H29` as the only plausible single-site break candidate

2. Manual confirmation pass
   - Artifact: `results/head_circuit/head_circuit_20260312_074809.json`
   - Config: `manual_heads=L5.H29,L3.H27,L19.H28,L24.H15,L17.H9,L7.H8,L8.H15,L5.H15`, `n_prompts=40`, `pair_prompt_limit=30`
   - Outcome: replicated `L05.H29` strongly, but pair summary still used an overly permissive superadditivity rule

3. Hardened manual confirmation pass
   - Artifact: `results/head_circuit/head_circuit_20260312_075244.json`
   - Same config as above, with `superadditive_margin=0.1`
   - Outcome: same single-head result, zero robust superadditive pairs

## Current read

- Clean recursive vs baseline gap is stable on base:
  - `clean_rv_gap = -0.15166`
- One early head now looks real:
  - `L05.H29`: `delta_rv = +0.16881`, `d = +1.0060`, `p = 0.001027`
  - Interpretation: replacing this head's recursive V-projection with the baseline version reliably weakens contraction
- Other tested heads are small:
  - next-largest absolute effect was `L03.H27`, `d = -0.2045`
- Tested pairs are not stronger than the single-head story:
  - all `L05.H29 + X` pairs were subadditive relative to the additive expectation
  - with the hardened rule, `n_superadditive_pairs = 0`

## Narrative implication

Base v0.1 head-to-head patching is no longer purely null. The strongest supported claim is:

`L05.H29` is a plausible early-gate carrier for recursive-specific geometry on base Mistral, while the tested top-site pairs do not show robust joint synergy.

That is materially different from the earlier "distributed-only/null-single-head" read and should supersede it for base-specific notes.

## Caveats

- Candidate discovery still depends on the older base head sweep artifact with `n=20`
- This experiment only patches head `V_PROJ` outputs; it is not a substitute for full path patching
- Pair tests covered only the top 8 manual sites and used `30` prompts in Phase 3

## Next highest-ROI follow-up

1. Run the required refreshed base head sweep at `n>=100`
2. Run full 32-layer base path patching
3. Check whether `L05.H29` remains high-salience in the refreshed sweep and whether full path patching localizes compatible flow through early-layer pathways
