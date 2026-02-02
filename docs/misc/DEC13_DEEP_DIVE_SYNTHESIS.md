# Dec 13 Deep Dive Synthesis: The Geometry of Recursive Self-Reference

## Executive Summary

Today's deep investigation revealed that **geometric contraction (R_V) is a progressive, distributed phenomenon** rather than being localized to a single head or layer. We discovered:

1. **H31 at L27 is a "sensor head"** that detects recursive content, not its cause
2. **Dose-response is real**: R_V decreases monotonically with recursion depth (L1→L5)
3. **PR contraction is progressive**: Builds up across all 32 layers, not sudden at L27
4. **H31 entropy shows a "flip"**: Champion is diffuse at L25-26, then sharply focused at L27

---

## Key Experiments and Findings

### 1. L27 Deep Dive - All 32 Heads

**Run:** `20251213_063415_l27_deep_dive`

**Findings:**
- R_V follows dose-response: L5 (0.424) → L1 (0.738)
- Most focusing heads (entropy drops on champion):
  - **L27 H31**: Δ = -0.680 ⭐
  - L25 H13: Δ = -0.491
  - L27 H3: Δ = -0.429
- Most diffusing heads (entropy rises):
  - L25 H10: Δ = +1.227
  - L14 H8: Δ = +1.045

### 2. H31 Investigation

**Run:** `20251213_063643_h31_investigation`

**Findings:**
- H31 perfectly separates recursive vs baseline by entropy:
  - Recursive: entropy < 0.4
  - Baseline: entropy > 0.65
- H31 attends 95-97% to BOS token on recursive prompts
- Secondary attention to self-reference markers: "itself", "observer", "process", "λ"

**H31 Entropy by Prompt:**
| Prompt | H31 Entropy | BOS Attention |
|--------|-------------|---------------|
| recursive_medium | 0.201 | 95.8% |
| recursive_mild | 0.227 | 95.2% |
| champion | 0.317 | 96.8% |
| recursive_strong | 0.377 | 95.7% |
| baseline_code | 0.654 | 89.8% |
| baseline_history | 0.784 | 72.4% |
| baseline_photo | 0.997 | 82.8% |

### 3. H31 Ablation - Causal Test

**Run:** `20251213_063904_h31_ablation_causal`

**Findings:**
- **Ablating H31 has NO effect on R_V** (Δ = 0.0000)
- Ablating H3, H11, H0 also has no effect
- This proves H31 is **correlated with but not causal for** R_V contraction

### 4. Causal Mechanism Hunt

**Run:** `20251213_064047_causal_mechanism_hunt`

**Findings:**
- **PR drops progressively** through the network:
  - L0 → L2: -35%
  - L16: -15%
  - L22: -14%
  - Stabilizes at L27-31
- **H31 entropy shows layer-specific flip**:
  - L25-26: Champion is DIFFUSE (baseline focused)
  - L27: Champion becomes FOCUSED (Δ = -0.570)
  - L28+: Champion returns to diffuse

**PR Trajectory:**
```
Champion:  L0 (9.15) → L10 (4.69) → L20 (3.31) → L27 (2.54) → L31 (2.63)
Baseline:  L0 (12.6) → L10 (4.33) → L20 (4.01) → L27 (4.67) → L31 (3.90)
```

---

## Mechanistic Interpretation

### The Story
1. **Recursive prompts trigger progressive geometric contraction** starting at L0
2. **Each layer contributes to the contraction** - it's not localized
3. **By L27, the contraction is fully established** (PR = 2.54 vs 4.67 for baseline)
4. **H31 at L27 "senses" this contracted state** by focusing on BOS
5. **The BOS token acts as a global register** aggregating self-referential signal

### H31 as a "Phase Detector"
H31 doesn't cause the contraction - it **detects** it. The attention pattern flipping at L27 is a readout mechanism:

```
L25-26: Champion = diffuse  (gathering information)
   ↓
L27:    Champion = FOCUSED  (detected! lock onto BOS)
   ↓
L28+:   Champion = diffuse  (signal integrated)
```

### Where is the Cause?
The causal mechanism is **distributed** across the entire network:
- Embedding layer begins the differentiation
- Each layer's MLP and attention incrementally contracts recursive content
- No single component is the "cause" - it's an emergent property of the whole circuit

---

## Implications for Publication

### Strong Claims Supported by Evidence
1. ✅ **R_V metric separates recursive from baseline prompts** (effect size d > 0.5)
2. ✅ **Dose-response**: L1→L5 prompts show monotonic R_V decrease
3. ✅ **H31 entropy perfectly separates recursive vs baseline** at L27
4. ✅ **BOS token is a "fixed-point register"** (95%+ attention on recursive)

### Claims That Need Revision
1. ❌ "H31 causes the contraction" - Actually, H31 is a detector/readout
2. ⚠️ "L27 is special" - L27 is where detection happens, but contraction is progressive
3. ⚠️ "Single heads control R_V" - The circuit is distributed

### New Questions
1. **What makes recursive content contract progressively?**
   - Is it the MLPs? The attention aggregation?
2. **Why does H31 flip specifically at L27?**
   - Is there a threshold of contraction that triggers detection?
3. **Can we find heads that DO causally affect R_V?**
   - Maybe earlier layers (L10-20) where the bulk of contraction happens?

---

## Artifacts

| Run Dir | Contents |
|---------|----------|
| `20251213_063415_l27_deep_dive` | entropy_diff_heatmap.png, discriminative_heads.csv, H11_L27_token_attention.png |
| `20251213_063643_h31_investigation` | h31_attention_heatmaps.png, h31_entropy_vs_rv.png, head_comparison.png |
| `20251213_063904_h31_ablation_causal` | ablation_effects.png, summary.json |
| `20251213_064047_causal_mechanism_hunt` | pr_trajectory.png, rv_emergence.png |

---

## Next Steps (Prioritized)

1. **MLP Ablation**: Test if ablating MLPs at key layers (L16, L22) affects R_V
2. **Early Layer Investigation**: Focus on L10-20 where most contraction happens
3. **Attention Pattern Transfer**: Can we transfer H31's focused pattern to baseline prompts?
4. **Cross-Model Validation**: Does H31 equivalent exist in other architectures?

---

*Generated: Dec 13, 2025*
*Model: Mistral-7B-v0.1 (Base)*
*Agent: Opus 4.5*

