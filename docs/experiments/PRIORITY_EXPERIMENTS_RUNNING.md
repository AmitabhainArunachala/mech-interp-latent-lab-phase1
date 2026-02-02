# Priority Experiments: Position-Specific & Sufficiency Tests

**Date:** January 5, 2025  
**Status:** Both experiments running in parallel

---

## Experiment 1: Position-Specific L0 Ablation

### Goal
Determine which token positions drive the L0 MLP effect.

### What We're Testing
- **BOS only** (position 0): Is L0 effect driven by first token?
- **First 4 tokens** (positions 0-3): Is it early-position processing?
- **Last 16 tokens** (matches R_V window): Is it the tokens we measure?
- **All tokens** (baseline): Current full-sequence ablation

### Expected Outcomes

**Scenario A: BOS-Driven**
- BOS-only ablation removes contraction → L0 is "early context classifier"
- First-4 ablation removes contraction → L0 processes early context
- Last-16 ablation has no effect → Not about measurement window
- **Interpretation:** L0 is a general early-position gate, not recursive-specific

**Scenario B: Token-Distributed**
- Only all-tokens ablation removes contraction → Diffuse effect
- Position-specific ablations have partial effects → Distributed across tokens
- **Interpretation:** L0 effect is distributed, weaker claim

**Scenario C: Measurement Window**
- Last-16 ablation removes contraction → L0 writes to tokens we measure
- BOS-only has no effect → Not about early processing
- **Interpretation:** L0 effect is specific to measurement window

### Parameters
- **Layer:** L0
- **Pairs:** 30
- **Expected duration:** ~30-45 minutes

### Log
`/tmp/position_specific_l0.log`

---

## Experiment 2: L0 Sufficiency Test (Denoising)

### Goal
Test if L0 MLP alone is **SUFFICIENT** to induce contraction, not just necessary.

### What We're Testing
1. Run model on **BASELINE prompt** (clean, no recursion)
2. Patch **ONLY L0 MLP output** from **RECURSIVE prompt** activations
3. Measure: Does R_V contract? Does behavior shift toward recursive?

### Expected Outcomes

**Scenario A: L0 is Sufficient**
- Patching L0 restores >50% of R_V gap → L0 alone sufficient
- Behavior shifts toward recursive → L0 drives recursive mode
- **Interpretation:** L0 is the complete story, no other components needed

**Scenario B: L0 is Partially Sufficient**
- Patching L0 restores 20-50% of R_V gap → L0 necessary but not sufficient alone
- Behavior partially shifts → L0 is one component among many
- **Interpretation:** L0 is necessary but needs other components (L1, L18-L20, etc.)

**Scenario C: L0 is Not Sufficient**
- Patching L0 restores <20% of R_V gap → L0 not sufficient alone
- Behavior doesn't shift → L0 is necessary but insufficient
- **Interpretation:** L0 is one piece of a larger mechanism

### Parameters
- **Layer:** L0
- **Pairs:** 30
- **Expected duration:** ~20-30 minutes

### Log
`/tmp/mlp_sufficiency_l0.log`

---

## What These Tests Tell Us

### Position-Specific Test Answers:
- **What kind of finding is L0?**
  - BOS-driven → "early context classifier"
  - Token-distributed → "recursive mode gate"
  - Measurement window → "writes to tokens we measure"

### Sufficiency Test Answers:
- **Is L0 the complete story?**
  - Sufficient → L0 alone drives contraction
  - Partially sufficient → L0 is primary but needs help
  - Not sufficient → L0 is one component among many

---

## Combined Interpretation

| Position Test | Sufficiency Test | Interpretation |
|---------------|------------------|----------------|
| BOS-driven | Sufficient | L0 is early-position gate, sufficient alone |
| BOS-driven | Not sufficient | L0 is early-position gate, needs other components |
| Token-distributed | Sufficient | L0 is recursive mode gate, sufficient alone |
| Token-distributed | Not sufficient | L0 is recursive mode gate, needs other components |
| Measurement window | Sufficient | L0 writes to measurement window, sufficient alone |
| Measurement window | Not sufficient | L0 writes to measurement window, needs other components |

---

## Status

**Both experiments running in parallel:**
- Position-specific: PID 6005
- Sufficiency test: PID 6162

**Monitor:**
```bash
ssh runpod-current 'tail -f /tmp/position_specific_l0.log'
ssh runpod-current 'tail -f /tmp/mlp_sufficiency_l0.log'
```

**Expected completion:** ~30-45 minutes

---

**These two tests determine whether L0 is:**
1. A general early-position gate (BOS-driven) vs recursive-specific gate
2. Sufficient alone vs needs other components


