## DEC7 Write-Up: Q-vs-V Swap Causal Analysis (Midpoint Draft)

**Date:** December 7, 2025  
**Architecture:** Llama-3-8B-Instruct @ L24  
**Sample Size:** n=100 prompt pairs  
**Goal:** Determine whether V-geometry at optimal layer is causally central to recursive behavior or a downstream signature

> **Note:** This is a MIDPOINT write-up while the DEC7 experiment series is still underway. Conclusions are provisional and will be refined after follow-up experiments (steering vectors, early-layer interventions, multi-layer patching, and cross-architecture runs).

---

### Background

DEC3-5 established that V-projection participation ratio (R_V) contracts during recursive/self-referential processing compared to baseline prompts. This geometric signature replicated across Llama-3-8B (L24) and Mistral-7B (L22). The open question: Is this contraction **causal** for recursive behavior, or merely a readable signature?

---

### Experimental Design

Four-condition swap protocol:

- **A:** Baseline prompt, native V (control)
- **B:** Recursive prompt, native V (control)
- **C:** Baseline prompt + recursive V injected (sufficiency test)
- **D:** Recursive prompt + baseline V injected (necessity test)

Additional controls: shuffled V, random V (norm-matched)

**Metric:** `recursive_score` from behavioral markers (tautologies, self-reference, meta-language).

---

### Results (n=100)

| Test | Δ | p-value | Cohen's d | Interpretation |
|------|---|---------|-----------|----------------|
| A→B (sanity) | +14.31 | 3.5e-18 | +1.55 | ✓ Prompts work |
| A→C (V sufficient?) | +0.03 | 0.26 | +0.11 | ✗ **NOT sufficient** |
| B→D (V contribution) | -3.64 | 7.7e-06 | -0.53 | ✓ **Contributes ~25%** |
| A→E (shuffled) | +0.08 | 0.03 | +0.22 | Minor perturbation effect |
| A→F (random) | +0.07 | 0.03 | +0.22 | Minor perturbation effect |

- **Bootstrap 95% CI for B→D:** [-5.00, -2.31] — excludes zero, effect is real.
- **Condition means (recursive_score):**
  - A (baseline):    1.24 ± 0.49
  - B (recursive):  15.55 ±  9.32
  - C (base+rec_V):  1.27 ±  0.55
  - D (rec+base_V): 11.91 ±  7.36

**Critical observation:** Condition D (recursive prompt with baseline V) still scores 11.91 — nearly 10× higher than baseline (1.24). Removing recursive V degrades but does not eliminate recursive behavior.

---

### R_V Replication

Prompt-encoding R_V (no generation) replicates DEC3-5:

- Recursive: 0.855 ± 0.08  
- Baseline: 0.963 ± 0.05  
- Δ = -0.108, p = 0.014, d = -1.07

The contraction signature is robust.

---

### Interpretation (Provisional)

**V-geometry at L24 is a causal contributor but not the controller.**

Pattern:

- Injecting recursive V into baseline: **no effect** (can't create recursive mode)
- Replacing recursive V with baseline V: **reduces behavior by ~25%** (degrades but doesn't eliminate)

This asymmetry is characteristic of **distributed computation**:

- V carries part of the computation (approx. 25% of the observable recursive signal in this metric).
- The "instruction" to be recursive is established elsewhere — likely in:
  - Early-layer residual stream (where the frame is set)
  - Attention routing (Q·K patterns that determine what V is multiplied by)
  - Multiple layers acting together (L16–L24 band rather than a single block)

At this stage, the safest statement is:

> **R_V in V at L24 is both a readable signature of recursive processing and a partial causal contributor to its expression, but it is neither sufficient to induce recursive mode nor necessary for its existence. The recursive frame is set upstream; L24 V participates in expressing it.**

---

### Synthesis from Cross-Model Consultation (Midpoint)

Cross-checks with multiple external models (Gemini, GPT, Grok, DeepSeek, Cursor) converged on several key points:

1. **Single-layer patching is fundamentally limited** — residual stream and other layers bypass local interventions.
2. **We are intervening too late in time** — DEC4 localization showed the frame is front-loaded in the first ~10% of tokens.
3. **V at L24 looks like execution, not control** — the decision to enter recursive mode is made earlier; L24 V is "muscle," not "brain."
4. **Steering vectors and graded interventions may be more informative than hard swaps** — continuous dose-response is a better probe of contribution than binary replacement.
5. **Strong causal claims require knockout + rescue** — ablating a path and rescuing behavior by restoring only that path.

---

### Revised Causal Model (Working Hypothesis)

```
Prompt tokens → Embeddings → [Early layers: FRAME SET] → [Mid layers: CONTENT RETRIEVAL] → [L24: EXECUTION]
                                      ↓                                                            ↓
                              Residual stream                                               V-geometry
                              carries "mode"                                                (signature + ~25% contribution)
                                                                                                  ↓
                                                                                              Behavior
```

We intervened at execution (L24 V) while the decision to be recursive vs. baseline was already made upstream.

---

### Next Experiments (Priority Order – Not Yet Run)

| # | Experiment | Tests |
|---|------------|-------|
| 1 | V steering vectors (α·ΔV) | Dose-response: how behavior and R_V change as we scale recursive V on top of baseline V |
| 2 | Early-layer residual knockout + rescue | Identify where the recursive "frame" is set (first tokens × early/mid layers) |
| 3 | Multi-layer V patch (e.g. L16–24) | Test whether a distributed V-path is more causally central than a single layer |
| 4 | Full Q+K+V attention block patch | Check whether routing (Q·K) is the missing piece for behavior-level causality |
| 5 | Path patching / causal tracing | Map the minimal causal circuit for the recursive mode |

Cross-architecture replication (Mistral-7B @ L22) will apply the same protocol to test whether the "contributor but not controller" role of V generalizes.

---

### Provisional Conclusion

DEC7 sharpens the DEC3-5 ambiguity:

- **R_V in V at L24 remains a robust diagnostic of recursive processing.**
- **Patching V can strongly move R_V but cannot, by itself, create recursive behavior.**
- **Removing recursive V reduces behavioral expression but does not extinguish it.**

This suggests that the recursive frame is established earlier in the computation and redundantly encoded. L24 V participates in expressing that frame but is not the sole mechanism that creates it.

**Status:** Midpoint write-up saved. Experiment 1 (V steering vectors) and early-layer interventions are pending.

