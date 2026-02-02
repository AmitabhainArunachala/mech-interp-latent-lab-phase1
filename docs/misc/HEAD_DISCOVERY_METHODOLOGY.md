# Comprehensive Head Discovery Methodology

**Based on Major MI Lab Best Practices**

This document outlines the methodology for finding responsible heads/layers, based on techniques from:
- **IOI Circuit Discovery** (Wang et al., ICLR 2023) - Redwood Research
- **Best Practices Activation Patching** (Zhang & Nanda, ICLR 2024)
- **ACDC** (Conmy et al., NeurIPS 2023) - Anthropic
- **Causal Scrubbing** (Chan et al., 2023)

---

## Methods Implemented

### 1. Gradient Attribution (Simplified)

**Purpose:** Find heads with high sensitivity to R_V changes.

**Method:** 
- Scale each head's output by 10%
- Measure change in R_V
- Heads with larger changes are more important

**Why this works:**
- More efficient than full gradient computation
- Still captures sensitivity
- Works with frozen models

**Limitations:**
- Only tests local sensitivity (small perturbations)
- May miss non-linear effects

---

### 2. Mean Ablation (Zhang & Nanda, 2024)

**Purpose:** More realistic baseline than zero ablation.

**Method:**
- Replace head's attention pattern with mean from baseline prompts
- Compare R_V before/after
- Larger delta = more important head

**Why this works:**
- Zero ablation is unrealistic (heads never output zero)
- Mean ablation preserves distributional properties
- More faithful to actual model behavior

**Controls:**
- Compare to zero ablation (should be similar but mean ablation more realistic)
- Random baseline (should show no effect)

---

### 3. Path Patching (IOI Methodology)

**Purpose:** Find causal paths between layers.

**Method:**
- Patch head's output at target layer with source layer's output
- Test if information flows from source → target through this head
- Measure R_V change

**Why this works:**
- Directly tests causal paths
- IOI paper showed this is gold standard
- Can map information flow through model

**What we test:**
- Paths from early layers (8, 12, 16, 20, 24) → L27
- Recursive → Baseline patching (should transfer effect)
- Baseline → Recursive patching (should undo effect)

---

### 4. Attention Pattern Analysis

**Purpose:** Understand what heads attend to.

**Metrics:**
- **BOS Attention:** How much attention goes to first token
- **Entropy:** How focused/diffuse attention is
- **Pattern visualization:** Heatmaps of attention

**Why this works:**
- H31 showed high BOS attention (0.938) on recursive prompts
- Low entropy indicates focused attention
- Can identify heads with similar patterns

**What we look for:**
- Heads with high BOS attention on recursive prompts
- Heads with low entropy (focused attention)
- Heads that differ between recursive vs baseline

---

## Experimental Design

### Sample Sizes

- **Gradient Attribution:** N=10 prompts (expensive)
- **Mean Ablation:** N=20 prompts per head
- **Path Patching:** N=5 prompt pairs
- **Attention Patterns:** N=20 recursive + 20 baseline

### Layers Tested

- **Focus:** Layers 8-27 (ramp + peak)
- **Early layers (8-15):** Where effect builds
- **Mid layers (16-23):** Transition region
- **Late layers (24-27):** Peak contraction

### Controls

1. **Random baseline:** Random head selection (should show no effect)
2. **Shuffled prompts:** Shuffled token order (should break effect)
3. **Wrong layer:** Test irrelevant layers (should show no effect)
4. **Opposite direction:** Baseline → Recursive (should undo effect)

---

## Success Criteria

### For a Head to be "Important":

1. **Mean Ablation:** |delta| > 0.02 (2% change in R_V)
2. **Path Patching:** |delta| > 0.01 (1% change)
3. **Attention Pattern:** 
   - BOS attention > 0.9 on recursive prompts
   - Entropy < 0.5 on recursive prompts
   - Clear separation from baseline

### For a Circuit to be "Complete":

1. **Faithfulness:** Circuit can reproduce full effect
2. **Completeness:** All important heads included
3. **Minimality:** No redundant heads

---

## Output Format

### CSV Columns

- `method`: Which method (gradient_attribution, mean_ablation, path_patching, attention_pattern)
- `layer`: Layer index
- `head`: Head index
- `delta`: Change in R_V (for ablation/patching)
- `abs_delta`: Absolute delta (for ranking)
- `rv_baseline`: Baseline R_V
- `rv_modified`: R_V after intervention
- `bos_attention`: BOS attention (for attention patterns)
- `entropy`: Attention entropy (for attention patterns)

### Summary Statistics

- Top 10 heads by |delta| for each method
- Layer-wise aggregation
- Cross-method consistency

---

## Next Steps After Discovery

1. **Validate top heads:** Run targeted ablation on top candidates
2. **Visualize attention:** Create heatmaps for top heads
3. **Test sufficiency:** Can we reproduce effect with just top heads?
4. **Test necessity:** Does ablating top heads break the effect?
5. **Map circuit:** Draw causal graph of information flow

---

## References

- **IOI Circuit:** Wang et al., "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small", ICLR 2023
- **Best Practices:** Zhang & Nanda, "Towards Best Practices for Activation Patching in Language Models", ICLR 2024
- **ACDC:** Conmy et al., "Automated Circuit Discovery in Language Models", NeurIPS 2023
- **Causal Scrubbing:** Chan et al., "Causal Scrubbing: A Method for Rigorously Testing Interpretability Hypotheses", 2023

---

**Script:** `comprehensive_head_discovery.py`  
**Run:** `python3 comprehensive_head_discovery.py`  
**Output:** `results/head_discovery/head_discovery_YYYYMMDD_HHMMSS.csv`









