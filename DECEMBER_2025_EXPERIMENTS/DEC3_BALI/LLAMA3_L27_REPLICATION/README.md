# LLAMA-3-8B L27 REPLICATION STUDY

## Purpose
Replicate Mistral-7B Layer 27 causal patching findings on Llama-3-8B-Instruct.

## Hypothesis
If the L27 geometric contraction effect is universal (not Mistral-specific), 
Llama-3-8B should show transfer efficiency >50% under identical methodology.

## Baseline (Mistral-7B results to beat)
- Transfer efficiency: 117.8%
- Cohen's d: -3.56
- p-value: < 10⁻⁶
- n: 45 prompt pairs
- Controls: All 4 behaved as predicted

## Methodology
Exact replication of `mistral_L27_FULL_VALIDATION.py` with only MODEL_NAME changed.

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | meta-llama/Meta-Llama-3-8B-Instruct | Changed |
| Target layer | 27 (84% depth) | Same |
| Early layer | 5 (15.6% depth) | Same |
| Window size | 16 tokens | Same |
| Prompt pairs | 45 | Same |
| Prompt bank | prompt_bank_1c | Same |
| Controls | Random, Shuffled, Wrong-layer (21), Main | Same |

## Success Criteria
- >50% transfer efficiency → Universal mechanism confirmed
- 20-50% → Partial replication, investigate
- ~0% → Architecture-specific (Mistral only)

## Date initiated
December 3, 2025

## Status
[ ] Pre-flight checks passed
[ ] Sanity check (1 pair) passed
[ ] Full validation (45 pairs) complete
[ ] Results analyzed
[ ] Comparison to Mistral documented

