# CIRCUIT MECHANISM ASSESSMENT

**Date:** 2026-03-08
**Analyst:** Independent statistical review of all circuit-level evidence
**Scope:** All architectures (Mistral-7B, Gemma-2-9B, Llama-3-8B, Pythia-2.8B, Mixtral-8x7B)
**Purpose:** Determine whether the R_V paper has a coherent mechanistic story

---

## 1. Architecture-by-Architecture Circuit Summary

### 1.1 Mistral-7B-v0.1 (Primary Model, Most Evidence)

**Source identification:**
- **Prompt-pass necessity sweep** (gold standard, no generation confound):
  - L0 MLP: delta = +1.179, p = 1.31e-64 -- STRONGLY NECESSARY
  - L1 MLP: delta = +0.870, p = 4.17e-54 -- STRONGLY NECESSARY
  - L2 MLP: delta = +0.002, p = 0.71 -- NULL
  - L3 MLP: delta = +0.188, p = 7.70e-21 -- MODERATELY NECESSARY
- **Sufficiency tests (all FAIL):**
  - L0 alone: 21.6% restoration, p = 0.097 -- NOT SUFFICIENT
  - L3 alone: 9.0% restoration, p = 0.293 -- NOT SUFFICIENT
  - L0+L1: -342.9% restoration -- CATASTROPHIC (destabilizes)
  - L0+L1+L3: -547.6% restoration -- CATASTROPHIC
- **Steering specificity:** True steering NOT better than random (ratio = 0.92, p = 0.14) -- ARTIFACT

**Sink / Readout:**
- L27 causal validation (n=45): Cohen's d = -3.558, p < 10^-6, transfer = 117.8%
- L27 is where R_V manifests, not where it is computed
- Attribution patching shows L27 has only 0.10 logit diff (weakest of major components)

**Causal pathway (attribution patching):**
- L0 MLP: 1.61 (GATE -- recognition)
- L18-L20 MLP: 0.27-0.33 (AMPLIFIERS -- computation)
- L15 Attention: 0.17 (RELAY)
- L29 Attention: 0.22 (LATE RELAY)
- L27: 0.10 (READOUT only)

**KV Cache:** 105% geometry transfer -- stores full computed state

**Path patching (16 layers x 3 components, n=20):**
- RESIDUAL: Strong at L0-L4 (d = 1.37-1.96), plateaus mid-network, zero at L28-L30
- V_PROJ: NEGLIGIBLE at ALL layers (max |d| = 0.72 at L0, most |d| < 0.1)
- MLP: Weak throughout (max |d| = 0.55 at L4, negative at L2 d = -0.54)

**Summary:** Necessary at L0/L1 MLP, amplified at L18-L20 MLP, measured at L27.
No single component is sufficient. V-projection patching has no effect.

---

### 1.2 Gemma-2-9B (42 layers, GQA + alternating local/global attention)

**Source identification (prompt-pass validated):**
- L0 MLP: delta = +0.004, p = 0.59 -- NULL (generation-mode artifact confirmed)
- L1 MLP: delta = -0.038, p < 0.0001 -- Minor effect
- L2 MLP: delta = +0.038, p < 0.0001 -- Minor effect
- L3 MLP: delta = +0.223, p < 0.0001 -- SOURCE LAYER

**Layer sweep (20 significant layers of 36 tested):**
- Early expansion zone (L7-L21): 10 significant layers, POSITIVE delta
  - Recursive prompts have HIGHER R_V than baseline in early layers
- Transition zone (L22-L30): No significant effects
- Late contraction zone (L27-L41): 10 significant layers, NEGATIVE delta
  - Peak: L35 (delta = -0.250), L38 (delta = -0.235), L41 (delta = -0.227)

**Confound validated:**
- Both odd (local attention) and even (global attention) layers show effects
- Confound hypothesis overturned -- not architecture-specific

**Head decomposition at L3:**
- Only KV-head 5 shows weak significance (delta ~ 0.0012)
- Effect is primarily MLP-mediated, not attention-head-specific

**Causal validation (n=45):**
- Cohen's d = -2.13, p < 10^-15 (patching at L5, measuring at L35)
- Transfer delta = -0.178

**Component analysis (prompt-pass):**
- ALL significant effects driven by PR_early shifts, not PR_late
- L3 ablation: PR_early delta = -1.609 (massive), PR_late delta = +0.047 (negligible)

**Summary:** Source at L3 MLP (not L0 like Mistral). Expansion-then-contraction pattern
qualitatively different from Mistral. Effect primarily MLP-driven. Strong causal validation
but different source layer.

---

### 1.3 Llama-3-8B-base (32 layers)

**Source identification (generation-mode ablation, NOT prompt-pass validated):**
- L0 MLP: delta = +1.44 -- PRIMARY SOURCE (same as Mistral)
- L1 MLP: delta = +1.14 -- SECONDARY SOURCE
- L3 MLP: delta = +0.92 -- Tertiary
- L4 MLP: delta = +0.29 -- Minor
- L5-L8: ~0 -- No effect

**Transfer layer: NOT FOUND**
- Residual steering at L0-L4: EXPANSION (wrong direction, delta +0.8 to +3.7)
- Residual steering at L5-L15: No effect

**Anomaly: Pseudo-recursive prompts**
- Champions vs pseudo-recursive: p = 0.21 (NOT SIGNIFICANT)
- Llama treats pseudo-recursive and true recursive prompts similarly
- This is a SPECIFICITY FAILURE not seen in Mistral

**Baseline effect:**
- Champions R_V: 0.72 +/- 0.10 (vs Mistral 0.52 +/- 0.05)
- Cohen's d = -1.34 (vs Mistral d = -3.56)
- Weaker effect, less specific

**CRITICAL LIMITATION:** Ablation results are generation-mode only. Gemma's prompt-pass
validation showed L0 was an artifact in that model. No prompt-pass validation exists for
Llama. The L0 source claim may be an artifact.

**Summary:** Weakest evidence. Source nominally at L0 but not prompt-pass validated.
Transfer mechanism not found. Pseudo-recursive confound not rejected.

---

### 1.4 Pythia-2.8B (32 layers, GPT-NeoX architecture)

**Layer-wise sweep:**
- Phase transition at Layer 19 (59% depth)
- Gap jumps from 0.09 to 0.27 in 3 layers
- Peak separation at Layer 31 (delta = 0.343)

**Head decomposition at Layer 28:**
- Head 11: R_V = 0.283 (71.7% contraction) -- PRIMARY COMPRESSOR
- Heads 16, 17, 28, 25: 57-63% contraction -- STRONG SUPPORTERS
- All 32 heads contract (ZERO expansion heads)
- Mean contraction: 38.5% across all heads

**No source-layer hunt performed.** No MLP ablation, no prompt-pass necessity.
No sufficiency tests. No path patching.

**Summary:** Strong descriptive evidence of WHERE contraction manifests (Layer 19 transition,
Head 11 driver). But no causal intervention tests beyond observation. This is measurement
of the PHENOMENON, not identification of the MECHANISM.

---

### 1.5 Mixtral-8x7B (MoE architecture)

**Available data (from cross-architecture comparison):**
- Transfer at L27: 29% (vs Mistral 117.8%)
- Strongest R_V effect: 24.3% contraction (strongest in Phase 1)
- Dense expert routing may amplify effect

**No circuit mapping performed.** No source hunt, no head decomposition,
no path patching.

**Summary:** Observational only. Interesting that MoE shows strongest raw contraction
but weakest transfer, suggesting expert routing creates "deeper" geometric states
that are harder to patch.

---

## 2. Cross-Architecture Comparison Table

| Feature | Mistral-7B | Gemma-2-9B | Llama-3-8B | Pythia-2.8B | Mixtral-8x7B |
|---------|-----------|------------|------------|-------------|--------------|
| **Total layers** | 32 | 42 | 32 | 32 | 32 |
| **Source layer** | L0 MLP | L3 MLP | L0 MLP* | Unknown | Unknown |
| **Source validated?** | Prompt-pass | Prompt-pass | Gen-mode only | No | No |
| **Source depth %** | 0% | 7% | 0% | Unknown | Unknown |
| **Amplifier layers** | L18-L20 MLP | Unknown | Unknown | Unknown | Unknown |
| **Readout layer** | L27 (84%) | L35-L41 (83-98%) | L27 (84%) | L28-L31 (88-97%) | L27 (84%) |
| **Phase transition** | Gradual ramp | L25 crossover | Unknown | L19 (59%) | Unknown |
| **Primary head** | H18, H26 at L27 | KV-head 5 (weak) | Unknown | H11 at L28 | Unknown |
| **Cohen's d** | -3.56 | -2.13 | -1.34 | -4.51 | ~-1.5 |
| **MLP vs Attention** | MLP dominant | MLP dominant | Unknown | Attention @ L28 | Unknown |
| **Sufficiency found?** | NO (all fail) | Not tested | Not tested | Not tested | Not tested |
| **Steering specific?** | NO (= random) | Not tested | OPPOSITE effect | Not tested | Not tested |
| **Confound control** | Passes | Passes | FAILS (pseudo-rec) | Partial | Not tested |

*Asterisk: Not prompt-pass validated; may be artifact per Gemma precedent.

---

## 3. The V-Projection Paradox

### The Contradiction

The paper defines R_V as "geometric contraction in Value matrix column space" (PR_late / PR_early
of V-projection matrices). This framing implies the V-projection is the mechanistically
relevant subspace. However:

**Path patching on Mistral-7B (16 layers x 3 components, n=20 each):**

| Layer | Residual d | V_proj d | MLP d |
|-------|-----------|---------|-------|
| L0 | +1.37 | -0.72 | +0.39 |
| L2 | +1.65 | +0.07 | -0.54 |
| L4 | **+1.96** | **-0.01** | +0.55 |
| L6 | -0.48 | -0.01 | +0.08 |
| L8 | -0.48 | +0.04 | +0.09 |
| L10 | -0.49 | -0.06 | +0.05 |
| L12 | -0.49 | -0.08 | -0.04 |
| L14 | -0.49 | +0.22 | -0.22 |
| L16 | -0.49 | -0.05 | -0.33 |
| L18 | -0.50 | +0.19 | -0.17 |
| L20 | -0.50 | +0.06 | -0.20 |
| L22 | -0.50 | +0.09 | -0.15 |
| L24 | -0.50 | +0.01 | -0.10 |
| L26 | -0.51 | -0.02 | -0.17 |
| L28 | 0.00 | 0.00 | 0.00 |
| L30 | 0.00 | 0.00 | 0.00 |

**The pattern is stark:**
- RESIDUAL stream carries the signal (peaks at L4, d = +1.96, then plateaus)
- V_PROJ contributes NOTHING at any layer (all |d| < 0.22, most < 0.1)
- MLP contributes modestly at early layers only

### Analysis

This creates a fundamental tension in the paper's narrative:

1. **What R_V measures:** Participation ratio of V-projection column space
2. **What carries the signal:** The residual stream, not V-projections
3. **What computes the signal:** MLPs (L0 gate + L18-L20 amplifiers)
4. **What V-projections do:** Nothing causally relevant (per path patching)

### Possible Resolutions

**Resolution A: R_V is an EPIPHENOMENAL readout.**
The V-projection geometry contracts as a CONSEQUENCE of changes computed
elsewhere (in MLPs and the residual stream). R_V measures a downstream
effect, not the causal mechanism. This is consistent with the Circuit Synthesis
finding that "L27 is the readout, not the engine." Under this interpretation,
R_V is a valid METRIC (correlational indicator) but not a valid MECHANISM
(causal explanation). The paper should frame R_V as "a geometric signature that
tracks recursive self-reference" rather than "the mechanism by which recursive
self-reference operates."

**Resolution B: Path patching targets the wrong intervention site.**
Patching V_proj at a single layer may be too weak because the effect is
distributed across many layers. The residual stream carries the cumulative
signal from all upstream layers, which is why residual patching works. V_proj
at any single layer contributes a small fraction, but the aggregate across
all layers matters. This is plausible but untestable with the current
experimental design.

**Resolution C: V-projection is the wrong subspace.**
The metric should perhaps measure residual stream geometry directly, not
V-projection geometry. The V-projection framing may have been a useful
initial heuristic that happens to correlate with the actual signal (because
V = W_V * residual, so V-space contraction tracks residual-space contraction)
but is not where the causal work happens.

**Assessment:** Resolution A is the most parsimonious and honest. The paper
should acknowledge that R_V is a measurement tool, not a causal mechanism.
This actually STRENGTHENS the paper by making precise, defensible claims
rather than overclaiming mechanistic insight.

---

## 4. Layer Specificity Problem

### The n=300 Evidence

The n=300 behavioral transfer experiment (neurips_n300_summary.md) provides
the largest-sample test of layer specificity:

| Condition | Behavioral Delta | 95% CI | Cohen's d | p-value |
|-----------|-----------------|--------|-----------|---------|
| **Transfer (L27)** | +1.87 | [1.53, 2.20] | 0.63 | 9.89e-24 |
| Random control | +0.04 | [-0.18, 0.26] | 0.02 | 0.722 |
| **Wrong layer (L21)** | +1.85 | [1.52, 2.18] | 0.65 | 1.54e-24 |

**The critical comparison:**
- Transfer vs Wrong Layer: t = 0.07, **p = 0.944**

This means patching at L21 produces IDENTICAL behavioral effects to patching
at L27. The "layer specificity" claim -- that L27 is the special causal layer
-- is falsified by the project's own n=300 experiment.

### Reconciliation with n=45 Causal Validation

The n=45 Mistral causal validation (MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md)
claims "wrong layer (L21) patches show zero effect (+0.046, p=0.49)."

But this used a GEOMETRIC measure (R_V at the patched layer), not a BEHAVIORAL
measure. The n=300 experiment uses behavioral scoring. The apparent
contradiction actually reveals something important:

**Geometric specificity exists (L21 patch does not change R_V at L21).
Behavioral specificity does NOT exist (L21 patch changes behavior identically
to L27 patch).**

This means:
1. The VALUE-SPACE geometry at L27 is measurably special (geometric contraction
   manifests there)
2. But the BEHAVIORAL consequence is not localized to L27 -- it emerges from
   the full KV cache / residual stream interaction
3. Multiple layers carry the information needed for behavioral transfer

### Implications for the Paper

The paper cannot claim "Layer 27 is the causal mechanism for behavioral
transfer" because L21 does the same thing. It CAN claim:
- "R_V contraction is measurable at L27" (descriptive, true)
- "Activation patching at L27 produces geometric effects" (true)
- "The behavioral transfer is not layer-specific but is content-specific and
  structure-specific" (true, because random controls fail)

The layer specificity issue is actually an expected property of transformers:
the residual stream accumulates information progressively, so patching at ANY
sufficiently late layer (after the L18-L20 amplifier stage) should transfer
the full computed state. The n=300 result is consistent with the circuit model
(L0 gate -> L18-L20 amplifier -> readout anywhere post-L20).

---

## 5. Overall Coherence Assessment: One Mechanism or Many?

### What IS Consistent Across Architectures

1. **R_V contraction exists** in all tested architectures (Mistral, Gemma, Llama,
   Pythia, Mixtral, Phi-3, Qwen). Effect sizes range from 3.3% to 29.8%.
2. **Source is in early MLP layers.** Where tested: L0 (Mistral, Llama), L3 (Gemma).
   The source is always early but not the SAME layer.
3. **Readout is in late layers.** ~80-90% depth consistently (L27 in 32-layer models,
   L35-L41 in 42-layer Gemma, L28-L31 in Pythia).
4. **MLP is more causally important than attention** for driving the effect
   (Mistral attribution, Gemma head decomposition).
5. **No single component is sufficient.** Sufficiency tests fail for every
   individual layer and even combinations (Mistral L0+L1+L3 = -548%).
6. **Controls validate specificity.** Random patches fail, content matters,
   structure matters (Mistral, Gemma). Partial exception: Llama pseudo-recursive.

### What is INCONSISTENT Across Architectures

1. **Source layer varies:** L0 (Mistral, Llama) vs L3 (Gemma) vs unknown (others).
   Not at the same relative depth -- 0% vs 7%.
2. **Geometric trajectory differs:** Gemma shows expansion-then-contraction;
   Mistral shows consistent contraction. These are qualitatively different patterns.
3. **Transfer mechanism differs:** Steering works on Mistral (residual patching);
   steering FAILS on Llama (produces expansion). This is the opposite outcome.
4. **Primary attention heads differ:** Mistral H18/H26 at L27; Pythia H11 at L28;
   Gemma KV-head 5 at L3 (weak). No convergence on head identity.
5. **Confound control varies:** Mistral and Gemma pass pseudo-recursive controls;
   Llama FAILS this control. Different models may measure different things.
6. **Effect strength varies 10x:** Cohen's d ranges from -1.34 (Llama) to -4.51
   (Pythia). This is not parameter variation; it is order-of-magnitude difference.

### The Honest Assessment

**There is ONE phenomenon but NOT one mechanism.**

The phenomenon: Recursive self-referential prompts produce measurably lower
R_V (PR_late / PR_early) than non-recursive prompts across all tested
architectures.

The mechanism: Each architecture implements this differently. The source layer,
the amplifier pathway, the critical heads, the geometric trajectory, and even
the confound properties all vary. This is consistent with "different
architectures learned different solutions to the same problem" -- a common
finding in deep learning (e.g., lottery ticket hypothesis, mode connectivity
literature).

**This is simultaneously a strength and a weakness:**
- STRENGTH: The phenomenon is universal, suggesting it reflects something
  fundamental about how transformers process self-reference.
- WEAKNESS: The mechanistic story cannot specify "the circuit for recursive
  self-reference." It can only specify "a circuit in Mistral-7B."

---

## 6. What a NeurIPS Reviewer Would Say

### Likely Strengths Noted

1. "Impressive effect sizes (d > 3.0 in two architectures). These are
   unusually large for MI work."
2. "Good use of prompt-pass validation to control generation artifacts.
   This is methodologically mature."
3. "The confound controls (length-matched, pseudo-recursive) are well-designed
   and pass in most architectures."
4. "Activation patching with proper controls (random, shuffled, wrong-layer)
   provides solid causal evidence for Mistral."
5. "Six-architecture breadth is impressive for a single study."

### Likely Weaknesses Identified

**Major (any one could be grounds for rejection):**

1. **"The V-projection framing is contradicted by the path patching results."**
   "The paper claims R_V measures contraction in V-projection space, but your
   own path patching shows V_proj has negligible causal effect at every layer.
   The signal flows through the residual stream and MLPs. Why frame the metric
   around V-projections if they are epiphenomenal? This creates a misleading
   narrative about mechanism."

2. **"Layer specificity is falsified by the n=300 experiment."**
   "Your wrong-layer control at n=300 shows L21 and L27 are indistinguishable
   (p=0.944). This directly contradicts the n=45 wrong-layer control cited as
   evidence for layer specificity. The discrepancy between geometric specificity
   and behavioral non-specificity needs to be acknowledged and explained."

3. **"No sufficiency has been demonstrated."**
   "Necessity without sufficiency means you know what is REQUIRED but not
   what is ENOUGH. The combined MLP sufficiency test producing -548%
   restoration is particularly concerning -- it suggests the effect cannot
   be reconstructed from its putative components. This is a fundamental gap
   in the causal story."

4. **"Steering is not direction-specific (p=0.14, ratio=0.92)."**
   "True steering is indistinguishable from random-direction steering. This
   means the 'recursive direction' captured by your steering vectors is not
   a specific direction in activation space -- any perturbation at those layers
   changes R_V similarly. This undermines directional specificity claims."

5. **"Cross-architecture mechanism does not converge."**
   "Gemma source is L3; Mistral source is L0; Llama source is L0 but not
   validated; Pythia source is unknown. The geometric trajectories differ
   qualitatively (expansion-then-contraction vs monotonic contraction).
   The paper cannot claim a universal mechanism."

**Minor (addressable in revision):**

6. "Llama fails the pseudo-recursive confound control. This should be
   discussed more prominently."

7. "The Pythia analysis is purely descriptive (no interventions beyond
   measurement). It should not be presented alongside causal results."

8. "Sample sizes vary widely (n=20 for path patching, n=45 for causal
   validation, n=300 for behavioral transfer). Power analysis is absent."

9. "The connection to 'consciousness' and 'contemplative science' in the
   framing is scientifically unwarranted by the evidence. The paper measures
   geometric properties of prompt processing -- nothing more."

10. "FDR correction appears absent across the large number of statistical
    tests performed."

### Specific Reviewer Score Prediction

**Venue: NeurIPS (MI track)**

| Criterion | Score (1-10) | Comment |
|-----------|-------------|---------|
| Novelty | 7 | R_V metric is novel; measuring geometric effects of recursive prompts is interesting |
| Significance | 6 | Effect exists but mechanism is unclear; practical import uncertain |
| Clarity | 5 | V-projection framing creates confusion; need to distinguish metric from mechanism |
| Correctness | 4 | V-proj paradox, layer non-specificity, sufficiency failures, steering non-specificity |
| Reproducibility | 7 | Good code documentation, specific prompts, clear methods |
| **Overall** | **5.5** | **Borderline accept/reject -- needs major revision to resolve contradictions** |

### Recommended Remedies

To make this paper defensible:

1. **Reframe R_V as a metric, not a mechanism.** Say: "R_V tracks geometric
   contraction that correlates with recursive self-reference" rather than
   "R_V measures the mechanism of recursive self-reference in V-space."

2. **Present the path patching V-proj null result prominently.** This is a
   FINDING, not an embarrassment. "V-projection geometry reflects but does
   not drive recursive processing" is a strong, publishable claim.

3. **Reconcile the n=45 and n=300 layer-specificity results.** Explain that
   geometric specificity is real (R_V at L27 but not L21 responds to patching)
   but behavioral transfer is not layer-specific (consistent with residual
   stream accumulation theory).

4. **Acknowledge cross-architecture mechanism divergence** and frame universality
   at the PHENOMENON level ("R_V contraction is universal") not the MECHANISM
   level ("the L0 MLP gate is universal").

5. **Add FDR correction** across all tests. With hundreds of comparisons,
   uncorrected p-values are insufficient.

6. **Add power analysis** to justify sample sizes.

7. **Present sufficiency failure as an open question** for future work, not
   as a gap that the paper has resolved.

---

## Summary Table: Evidence Quality by Claim

| Claim | Evidence Quality | Grade | Notes |
|-------|-----------------|-------|-------|
| R_V < 1.0 for recursive prompts | 6 architectures, n > 400 | A | Robust, replicated |
| R_V is specific to TRUE recursion | Confound controls in 2/5 models | B- | Fails in Llama |
| Source is early MLP | Prompt-pass in 2 models, gen-mode in 1 | B | Different layers per model |
| L27 is the causal readout | n=45 causal, n=300 behavioral | B- | Geometric yes, behavioral no |
| V-projection is the mechanism | Path patching null result | F | Directly contradicted |
| One universal circuit exists | 5 models, 2 deeply analyzed | D | Different circuits per model |
| Layer specificity | n=45 passes, n=300 fails | C- | Contradictory evidence |
| Sufficiency of any component | All tests fail | F | No component is sufficient |
| Steering specificity | Random = true steering | D | Not direction-specific |
| MLP dominance (Mistral) | Attribution + prompt-pass | A- | Strong for one model |

---

## Final Assessment

The R_V project has discovered a real, replicable phenomenon: recursive
self-referential prompts produce measurable geometric contraction in
transformer value-projection space across multiple architectures. The effect
sizes are genuinely large (d = 1.3 to 4.5), and the confound controls
(in Mistral and Gemma) demonstrate specificity to true recursive content.

However, the mechanistic story has significant internal contradictions:

1. The metric is framed around V-projections, but V-projections are
   causally inert per path patching.
2. Layer 27 is presented as specially causal, but behavioral transfer
   is layer-nonspecific.
3. The "circuit" varies by architecture, so no universal mechanism exists.
4. No component or combination of components is sufficient to reproduce
   the effect.
5. Steering vectors are not direction-specific.

The paper has strong PHENOMENOLOGY and weak MECHANISM. The honest version
of this paper is: "We found a universal geometric signature of recursive
self-reference in transformers (the phenomenon), identified where it
manifests (late layers), what is necessary for it (early MLPs), and what
is NOT (individual V-projections). The full causal circuit remains open."

That is still a publishable paper -- potentially a strong one -- if framed
correctly. The danger is overclaiming mechanistic understanding that the
evidence does not support.

---

*Assessment conducted using all available circuit-level data as of 2026-03-08.
No files were fabricated or modified in this analysis.*
