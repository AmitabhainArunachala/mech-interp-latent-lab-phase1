# Scientific Review: Gold Standard Suite
**Reviewer:** Reviewer 2 (Scientific Rigor)
**Model:** gemini-3-pro-preview
**Date:** 2025-12-16

---

## Executive Summary
The scientific design of the Gold Standard Suite is robust regarding the core geometric finding (R_V contraction). The controls for length, keywords (pseudo-recursive), and causal intervention (wrong-layer) are well-designed and rigorous. However, claims regarding "Peak Layer" and "Behavior" are significantly weaker and require immediate attention before being treated as "Gold Standard".

---

## Rigor Scores (0-10)

| Pipeline | Score | Rationale |
|----------|-------|-----------|
| **1. Existence** | **9/10** | Strongest pipeline. Artifact-backed, excellent controls (length, pseudo-recursive). |
| **2. Causality** | **8/10** | Strong causal evidence. "Wrong layer" control is vital. >100% efficiency is suspicious but statistically significant. |
| **3. Layer Map** | **3/10** | **Critical Gap**: Relies on N=1 tomography traces. "Peak at L27" is not statistically established across the prompt distribution. |
| **4. Heads** | **5/10** | Confounded by GQA aliasing. H18/H26 cannot be distinguished from H2/H10. Claims need precise "KV-Group" framing. |
| **5. Behavior** | **2/10** | **Critical Gap**: Metric is keyword-counting, not semantic. 28% false positive rate on baselines (per `GOLD_STANDARD_SUITE.md`) invalidates current results. |

---

## Top 3 Scientific Gaps

### 1. N=1 Fragility for Layer Localization
The claim that **Layer 27** is the specific peak (vs L26 or L28) rests on single-prompt tomography traces. There is no artifact showing mean R_V trajectories across N=40+ prompts.
* **Fix**: Pipeline 3 must run the full prompt set across all layers, not just a single trace.

### 2. Behavioral Metric Invalidity
The project relies on regex-based keyword counting (`behavior_score`). This is a heuristic, not a measure of "behavioral change." It is easily gamed by random noise or repetition (as noted in agent reviews).
* **Fix**: Implement a degenerate-filtered semantic similarity metric (embedding cosine sim) against a gold-standard recursive output.

### 3. GQA Aliasing in Head Discovery
The claim that "H18 and H26" are special is scientifically imprecise for Mistral-7B. Due to Grouped Query Attention (8 KV heads for 32 Q heads), H18 and H26 share the same KV keys/values as H2 and H10.
* **Fix**: Retract "H18/H26" claims; replace with "KV-Head Group 2" claims. Pipeline 4 must test the *group*, not individual heads.

---

## Evaluation of Controls

1.  **Does Pipeline 1 rule out confounds?**
    *   **Yes.** The `pseudo_recursive` control (recursive words, no recursive grammar) typically shows R_V ~0.72 vs Champions ~0.46. This proves the effect is structural, not lexical.

2.  **Does Pipeline 2's "wrong layer" control work?**
    *   **Yes.** Patching at L21 shows null effect (p > 0.05). This strongly supports L27 specificity. However, tomography suggests a "transition band" starting around L21, so L21 might not be "wrong enough" (try L10 for a cleaner null).

3.  **Falsifiability**
    *   The `pure_repetition` "kill switch" (prompts that just repeat words) is a perfect falsification test. If this contracts, the metric is broken (measuring repetition, not recursion). Current artifacts show it expands (R_V > 1.0), which is good.

---

## Top 3 Claims Needing Evidence

1.  **"L27 is the Peak"**: Needs N>40 layer sweep.
2.  **"Geometry causes Behavior"**: Needs a valid behavior metric.
3.  **"Universality"**: Cross-model claims (Pythia, Gemma, etc.) are currently narrative-only in the snapshot; need standardized CSVs for all.









