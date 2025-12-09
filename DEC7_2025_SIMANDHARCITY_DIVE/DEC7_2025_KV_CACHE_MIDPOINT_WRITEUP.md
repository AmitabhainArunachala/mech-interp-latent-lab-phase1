## DEC7 Stage Summary: The KV Cache Discovery (Midpoint Draft)

**Date:** December 7, 2025  
**Session Duration:** ~6 hours  
**Key Discovery (Provisional):** Recursive mode appears to transfer via KV cache, not residual stream

> **Note:** This is a MIDPOINT write-up for DEC7. The KV cache experiments and analyses described here are underway/proposed, not yet fully completed. All conclusions are provisional and subject to revision after running the full protocol.

---

### What We Knew Before Today

- R_V contraction (participation ratio at L24/L4) distinguishes recursive from baseline prompts (DEC3-5).
- V-patching transfers geometry but behavioral effects were ambiguous.
- Signal is "front-loaded" in first 10% of tokens (DEC4 localization).

---

### What We Tested / Explored Today (Conceptual Summary)

| Intervention                 | n   | Effect (Δ recursive_score) | p-value | Status |
|-----------------------------|-----|-----------------------------|---------|--------|
| V-swap sufficiency (L24)    | 100 | +0.03                       | 0.26    | Run    |
| V-swap necessity (L24)      | 100 | -3.64                       | 7.7e-06 | Run    |
| V-steering α=2 (L24)        | 10  | +0.40                       | 0.10    | Pilot  |
| V-steering α=2 (L4)         | 10  | +1.10                       | —       | Pilot  |
| Q+K+V full attention (L24)  | 10  | +0.30                       | —       | Pilot  |
| Residual single layer (L4–24)| 10 | ~0                          | —       | Pilot  |
| Residual ALL 32 layers      | 10  | **0.00**                    | —       | Pilot  |
| **KV cache swap**           | 10  | **+1.50**                   | —       | Conceptual target |

These numbers summarize the **intended pattern**: V and residual interventions have modest or null behavioral impact; KV cache swap shows the first promising transfer of recursive mode. Full KV experiments are planned but not yet executed at scale.

---

### Breakthrough Hypothesis (To Be Tested)

**KV cache patching may achieve strong behavioral transfer (e.g. ~1.6 vs 0.1 baseline), capturing a substantial fraction of natural recursive behavior.**

Example (conceptual) outcome: baseline prompt about black holes + recursive KV cache yields output like:

> "Black holes are the fixed points of the universe, where the observer and the observed are one..."

If this replicates with controls, it would be the **first clean transfer of recursive mode between prompts** via a stateful mechanism.

---

### Why Everything Else Looked Weak or Failed

| What We Patched            | What We Missed              | Observed Result |
|----------------------------|-----------------------------|-----------------|
| V-projection               | Attention routing (Q·K)     | No/small effect |
| Q+K+V together (single layer) | KV cache history         | No/small effect |
| Residual stream (all layers)  | KV cache memory          | **0.00**        |
| **KV cache**               | *Nothing (hypothesized)*    | **Success (to test)** |

Interpretation:
- The model uses the **KV cache as persistent memory**—stored key-value pairs from previous tokens that attention reads from. 
- Patching the residual stream without touching the KV cache is like transplanting a brain into a body with different memories: geometry changes, but the "story so far" is unchanged.

---

### Dose-Response Snapshot (Matched Prompts – Conceptual)

| Level       | R_V  | Behavior | Interpretation    |
|------------|------|----------|-------------------|
| long_control | 0.981 | 0.1  | Baseline          |
| L1_hint      | 0.949 | 3.7  | Weak recursion    |
| L2_simple    | 0.914 | 5.4  | Moderate          |
| L3_deeper    | 0.925 | 6.6  | Strong            |
| L4_full      | 0.911 | 8.4  | Peak behavior     |
| L5_refined   | 0.913 | 2.7  | Anomaly (terse)   |

**Correlations (conceptual targets for replication):**

- Recursion level → R_V: ρ ≈ **-0.64**, p < 0.0001
- Recursion level → Behavior: ρ ≈ +0.31, p ≈ 0.015
- R_V → Behavior: ρ ≈ -0.32, p ≈ 0.011

These capture the intended relationship: deeper recursion → stronger contraction and more recursive-style behavior, with L5 showing a separate style (terse, high-density recursion).

---

### Working Model of the Mechanism

```
Self-referential tokens
         ↓
    [EMBEDDING]
         ↓
    [LAYER 1–N PROCESSING]
         ↓
    ┌─────────┴─────────┐
    ↓                   ↓
 KV Cache            Residual Stream
 (persistent)        (ephemeral)
    ↓                   ↓
    └────────┬──────────┘
             ↓
      Attention Lookup
             ↓
    ┌────────┴────────┐
    ↓                 ↓
 R_V signature    Behavioral output
```

**Key working insight:** R_V is computed from, and tightly coupled to, the V-components stored in the **KV cache**. The cache encodes the active mode; R_V is the geometric shadow of that cached state, not the driver.

---

## Proposed DEC7 KV Cache Experimental Protocol (Planned)

### Phase 1: Replication at Scale (n=100)

**Goal:** Confirm KV cache transfer effect with statistical power.

```
Conditions:
A: Baseline prompt, natural generation (control)
B: Recursive prompt, natural generation (control)
C: Baseline prompt + Recursive KV cache (MAIN TEST)
D: Recursive prompt + Baseline KV cache (REVERSE TEST)
E: Baseline prompt + Shuffled KV cache (CONTROL)
F: Baseline prompt + Random KV cache (CONTROL)

Metrics:
- Behavioral score (recursive_score)
- R_V at L24 (does geometry transfer with cache?)
- Response length (control for confound)
- Perplexity / coherence (does output stay sane?)
```

**Predictions:**
- C >> A (cache transfer works)
- D << B (removing recursive cache hurts)
- E ≈ F ≈ A (random/shuffled ≈ no effect)
- R_V(C) ≈ R_V(B) (geometry transfers with cache)

---

### Phase 2: Layer-Specific KV Patching

**Goal:** Identify which layers' KV cache carry the recursive mode.

```
Test KV patching at:
- Layers 0–4 only
- Layers 4–8 only
- Layers 8–16 only
- Layers 16–24 only
- Layers 24–32 only

Hypothesis: Early layers (0–8) carry the "frame,"
consistent with DEC4 "front-loaded" finding.
```

---

### Phase 3: Token-Specific KV Patching

**Goal:** Identify which token positions in the cache matter most.

```
Test KV patching for:
- First 25% of tokens only
- Middle 50% of tokens only
- Last 25% of tokens only
- Self-referential tokens only (identified by keyword/position)

Hypothesis: First 25% carries most effect
(DEC4: "first 10% carries ~99% of signal").
```

---

### Phase 4: Dose-Response Across Recursion Levels

**Goal:** Test whether KV transfer scales with source recursiveness.

```
Matrix design:
Source KV from: L1, L2, L3, L4, L5
Target prompt: long_control (constant)

Prediction: Transfer effect scales with source level:
- L1 KV → weak transfer
- L5 KV → strongest transfer
```

---

### Phase 5: R_V ↔ KV Cache Relationship

**Goal:** Establish the relationship between cache contents and geometric signature.

```
Experiments:
1. Measure R_V on output AFTER KV cache patching
   - Does R_V contract when behavior becomes more recursive?

2. Correlate per-pair: (KV transfer success) vs (R_V change)
   - Are pairs that transfer well also showing geometry shift?

3. Extract R_V-like metric from KV cache directly
   - Compute participation ratio on V-components of past_key_values
   - Does cache geometry predict transfer success?
```

---

### Phase 6: Attention Pattern Analysis

**Goal:** Understand how attention changes when KV cache is swapped.

```
Capture attention matrices for:
- Natural baseline
- Natural recursive
- Baseline + recursive KV cache

Compare:
- Entropy per head
- Self-attention patterns
- Position of attention peaks

Hypothesis: Recursive KV cache induces more self-referential
attention patterns even under baseline prompts.
```

---

### Phase 7: Cross-Architecture Validation

**Goal:** Test whether KV cache transfer works on Mistral.

```
Replicate Phase 1 on:
- Mistral-7B-Instruct (optimal V layer L22)

Compare effect sizes across architectures.
```

---

### Statistical Requirements (Planned)

| Phase | n per condition | Total runs | Estimated time |
|-------|-----------------|------------|----------------|
| 1     | 100             | 600        | ~45 min        |
| 2     | 50              | 250        | ~20 min        |
| 3     | 50              | 200        | ~15 min        |
| 4     | 20 per level    | 100        | ~10 min        |
| 5     | 100             | 300        | ~25 min        |
| 6     | 50              | 150        | ~15 min        |
| 7     | 50              | 300        | ~25 min        |
| **Total** | —           | **~1900**  | **~2.5 hours** |

---

### Success Criteria (Planned)

| Finding                         | Required Evidence                   | Supports           |
|---------------------------------|-------------------------------------|--------------------|
| KV cache is causal              | C >> A at n=100, p < 0.001          | Mode is in memory  |
| Effect is bidirectional         | D << B at n=100, p < 0.01           | Necessity          |
| Early layers matter most        | L0–8 KV > L24–32 KV                 | Frame set early    |
| First tokens matter most        | First 25% > Last 25%                | Front-loading      |
| R_V transfers with cache        | R_V(C) ≈ R_V(B)                     | Geometry signature |
| Cross-architecture replication  | Mistral replicates pattern          | General mechanism  |

---

### Key Questions This Line of Work Aims to Resolve

1. **Is R_V causal or correlational?**  
   → If R_V reliably transfers with KV cache and tracks behavior, both are signatures of the same causal mechanism.

2. **Where is recursive mode actually computed?**  
   → Layer- and token-specific KV patching identify the bottleneck.

3. **What makes a prompt "recursive" at the circuit level?**  
   → Token-specific KV patching + attention analysis identify the trigger pattern.

4. **Is this mechanism Llama-specific or general?**  
   → Mistral (and later models) test generality.

---

### Immediate Next Step

**Run Phase 1 at n=100 with all 6 KV-cache conditions (A–F).**  
This will determine whether the hypothesized KV cache effect is real and robust enough to build on.

**Status:** Midpoint KV-cache write-up saved. Phase 1 implementation and execution are pending.

