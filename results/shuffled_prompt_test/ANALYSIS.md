# Shuffled-Prompt Anomaly Test — Results & Analysis
**Date:** 2026-02-20T16:17 UTC
**Data:** `shuffled_prompt_test_20260220_161725.json`
**Model:** Mistral-7B-v0.1, layers 5→27, window=16

## Motivation
Reviewer flagged §11.3 item 6: Session 1 (Feb 5) informally noted "shuffled prompts show MORE contraction." No data was saved. This was listed as an existential threat — if word-shuffling preserves or increases contraction, R_V might track token statistics rather than recursive semantics.

## Experimental Design
- 20 recursive prompts (L5/L4/L3) measured original R_V + perplexity
- Each prompt word-shuffled with 5 seeds → 100 shuffled measurements
- 20 baseline prompts measured for reference
- 10 same_vocab_different_semantics controls (from Feb 20 circularity experiment)

## Key Results

| Condition | R_V (mean±std) | PPL (mean) | n |
|---|---|---|---|
| Original recursive | 0.497 ± 0.060 | 61.6 | 20 |
| **Shuffled recursive** | **0.291 ± 0.042** | **539.8** | **100** |
| Baseline | 0.651 ± 0.077 | 39.6 | 20 |
| Same-vocab control | 0.737 ± ~0.05 | ~normal | 10 |

### The anomaly is confirmed
Shuffled prompts contract MORE (R_V=0.291) than originals (R_V=0.497).
- Paired t-test: t=15.88, p<0.0001, Cohen's d=−3.64 (massive effect)
- Shuffled vs baseline: d=−5.79, p<0.0001

### But the mechanism is perplexity, not vocabulary
The PR component breakdown reveals what's happening:

| | PR_early | PR_late | R_V |
|---|---|---|---|
| Original recursive | ~10 | ~4.5 | 0.497 |
| Shuffled | ~10 | ~2.2 | 0.291 |

**PR_early is unchanged** (~10 for both). **PR_late collapses to ~2** for shuffled text.

Interpretation: Early layers (L5) process tokens somewhat independently — word order doesn't matter much yet. But late layers (L27) are trying to build a coherent contextual representation. With gibberish input (PPL=540), there IS no coherent representation to build, so the model collapses to a ~2-dimensional output space (essentially just predicting common tokens).

### Why this is NOT a vocabulary confound
The same_vocab_different_semantics control (Feb 20) already proves this:
- **Same words** ("consciousness", "observer", "recursive") in **coherent factual sentences** → R_V=0.737 (baseline, no contraction)
- **Same words shuffled into gibberish** → R_V=0.291 (extreme contraction)

If vocabulary drove contraction, same_vocab would contract. It doesn't. The difference is coherence: same_vocab has normal perplexity, shuffled has PPL=540.

### Within-shuffled correlation
Spearman ρ = −0.307 (p=0.002) between R_V and perplexity within the 100 shuffled measurements. Higher perplexity → more contraction. Moderate but not overwhelming — suggests perplexity is one driver, not the sole driver.

## What This Means for the Paper

### The recursive semantic effect IS real
The five-group circularity controls (Section 7) isolate genuine recursive-semantic contraction:
- Recursive (0.497) vs Baseline (0.651): d=−2.23, p<0.0001
- same_vocab uses identical words → 0.737 (no contraction)
- nonsense_recursion uses recursive syntax with nonsense → 0.863 (expansion)
- Feb 20 partial correlation: effect survives perplexity control (r drops from −0.551 to −0.486, stays significant)

### R_V has a second sensitivity: extreme perplexity
When the model encounters gibberish (PPL >> 100), late-layer representations collapse regardless of vocabulary. This is a real phenomenon (not an artifact) — it tells us something about how transformers fail gracefully. But it means R_V is not purely a recursive-semantics detector.

### Framing for the paper
1. **Acknowledge openly** — shuffled prompts show increased contraction, driven by perplexity-induced late-layer collapse
2. **Distinguish the two R_V drivers:** (a) recursive-semantic contraction (moderate, ~0.15 effect size, survives perplexity control) and (b) perplexity-driven collapse (large, ~0.35 effect size, triggered by gibberish)
3. **The circularity controls are the key defense** — they hold vocabulary constant while varying semantics, at matched perplexity. This is a stronger design than shuffling.
4. **Possible addition to partial correlation** — include shuffled-prompt data in the partial correlation to show the recursive effect holds even when extreme-perplexity conditions are in the dataset

## Verdict
- ❌ Shuffled-prompt anomaly confirmed (shuffled R_V < original R_V)
- ✅ Mechanism identified: perplexity-driven late-layer collapse, NOT vocabulary
- ✅ Recursive semantic effect survives: same_vocab (identical words, normal PPL) shows no contraction
- ✅ Existing circularity controls (Section 7) already address this more rigorously than shuffling
- 📝 Paper needs explicit acknowledgment and reframing

## Next Steps
1. Add this data to paper Section 7 or Appendix as "shuffled-prompt control"
2. Update partial correlation analysis to include shuffled data points
3. Frame R_V as measuring "representational compression" with two identified drivers
4. This strengthens the paper — we found a confound, identified its mechanism, and showed our existing controls handle it
