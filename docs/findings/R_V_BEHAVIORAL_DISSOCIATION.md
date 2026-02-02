# R_V and Behavioral Transfer Are Dissociable

**Date:** 2026-01-25
**Status:** CONFIRMED FINDING
**Significance:** HIGH - Changes causal mechanism interpretation

---

## Summary

KV cache patching produces strong behavioral transfer (d=2.494) but does NOT transfer the R_V geometric signature. This dissociation has important implications for the mechanism of behavioral change.

---

## Key Results

### Experiment 1: Output R_V (Post-Generation)

From `gemma_full_validation/summary_20260125.json`:

| Condition | Mean R_V | SD |
|-----------|----------|-----|
| Champion prompt | 0.567 | - |
| Baseline (input) | 0.971 | - |
| Patched (output) | 0.993 | - |
| **d (patched vs baseline R_V)** | **0.11** | NS |

**Finding:** Patched output R_V is indistinguishable from baseline R_V. The geometric signature is NOT inherited.

### Experiment 2: R_V Trajectory During Generation

From `gemma_rv_during_generation.json`:

| Condition | Mean R_V | SD | Range |
|-----------|----------|-----|-------|
| Champion prompt | 0.643 | - | - |
| Baseline generation | 0.815 | 0.128 | [0.565, 1.135] |
| Patched generation | 0.784 | 0.105 | [0.560, 1.026] |
| **Gap from champion** | | | |
| - Baseline | 0.172 | - | - |
| - Patched | 0.141 | - | - |

**Finding:** Patched generation shows slightly lower R_V during generation (~0.03 closer to champion), but the effect is small and not statistically significant. R_V does NOT track closely with behavioral changes.

---

## The Dissociation

```
BEHAVIORAL TRANSFER                    R_V TRANSFER
──────────────────────────────────────────────────────────────
Markers: 0.3 → 27.4 (91x)             R_V: ~0.97 → ~0.99 (NS)
EOS: 30% → 0%                          No change
Cohen's d = 2.494***                   Cohen's d = 0.11 (NS)
──────────────────────────────────────────────────────────────
STRONG EFFECT                          NO EFFECT
```

---

## Mechanistic Implications

### Original Hypothesis (FALSIFIED)

```
Champion → low R_V → patch KV → transfer low R_V → behavioral markers
```

This would predict:
- Patched output should have R_V closer to champion (~0.6)
- R_V during generation should track behavioral markers

Neither is observed.

### Revised Hypothesis

```
Champion → attention patterns encoding self-referential vocabulary → patch KV
                                                                      ↓
                         behavioral markers ←─── direct vocabulary priming
                         (no R_V mediation)
```

The KV cache contains both:
1. **Geometric information** (captured by R_V) - NOT transferred
2. **Attention pattern information** (vocabulary bias) - IS transferred

KV patching appears to transfer the **attention priming** without the **geometric contraction**.

---

## Why This Matters

### For R_V Research

R_V measures something real (geometric contraction during self-referential processing), but this signature:
- Is **not stored persistently** in the KV cache in a transferable way
- Does NOT propagate into generated text
- May be **epiphenomenal** to the behavioral effect

### For Behavioral Transfer

The behavioral transfer is robust (d=2.5) but likely operates through:
- Direct **vocabulary/attention bias transfer** from KV cache
- NOT through geometric attractor installation

This is closer to the reviewers' "content leakage" concern than we initially admitted.

### For Publication

This is actually a **stronger finding** than "causal loop closed":

> "KV cache patching produces robust behavioral transfer (d=2.5) through attention priming mechanisms, but the R_V geometric signature is not preserved. This dissociates behavioral effects from geometric contraction, suggesting R_V captures prompt processing dynamics rather than generative attractors."

---

## Next Steps

1. **Attention pattern analysis**: What attention heads are most affected by patching?
2. **Vocabulary embedding analysis**: Is the effect purely lexical?
3. **Causal intervention on R_V**: Can we patch ONLY R_V (via V_PROJ) and test behavioral effect?
4. **Cross-model validation**: Does this dissociation hold for Mistral?

---

## Files

- `results/gemma_full_validation/summary_20260125.json` - Full validation stats
- `results/gemma_rv_during_generation.json` - R_V trajectory data
- `gemma_rv_during_generation.py` - Trajectory measurement script
- `gemma_full_validation_v2.py` - Validation script

---

*Documented 2026-01-25*
