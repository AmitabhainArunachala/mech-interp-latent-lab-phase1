# Measurement Contract

**Status:** LOCKED  
**Version:** 1.1  
**Date:** December 16, 2024

---

## Purpose

This document defines the **canonical measurement contract** for all experiments. It ensures that "same prompts → same numbers" is mechanical, not accidental.

---

## Geometry Contract

### R_V Metric Definition (Static / Prompt-Pass)

$$R_V = \frac{PR(V_{late})}{PR(V_{early})}$$

Where:
- **PR** (Participation Ratio) = $\frac{(\sum \lambda_i^2)^2}{\sum (\lambda_i^2)^2}$
- $\lambda_i$ are singular values from SVD of V-projection window
- **Early layer:** 5 (after initial processing)
- **Late layer:** `num_layers - 5` (typically 27 for 32-layer models)
- **Window:** Last W=16 tokens of the **prompt**

### Dynamic R_V Metric Definition (Temporal / Rolling)

$$R_V(t) = \frac{PR(V_{late}(t))}{PR(V_{early}(t))}$$

Where:
- Measured at generation step `t`
- **Window:** Last W=16 tokens of the **current sequence** (prompt + `t` generated tokens)
- **Warning:** $R_V(t=0)$ should match Static $R_V$, but $R_V(t>0)$ includes generated content.

### Standard Parameters

- **Early layer:** 5 (fixed)
- **Late layer:** Model-dependent (`num_layers - 5`)
- **Window size:** 16 tokens (fixed)
- **Contraction threshold:** R_V < 0.8

### NaN Handling Rules

1. **Short prompts:** If prompt length < window_size, return NaN
2. **Degenerate SVD:** If total variance < 1e-10, return NaN
3. **Zero PR:** If PR_early == 0, return NaN
4. **Invalid tensors:** If V-projection capture fails, return NaN

### Implementation

- **File:** `src/metrics/rv.py`
- **Function:** `compute_rv(model, tokenizer, text, early=5, late=27, window=16, device="cuda")`
- **Validation:** All NaN cases must be documented and handled consistently

---

## Generation Contract

### Tier 1: Reproducibility (Deterministic)

- **Temperature:** 0.0 (greedy decoding)
- **Seed:** Fixed (default: 42)
- **Purpose:** Ensure exact reproducibility
- **Use case:** Primary measurements, validation

### Tier 2: Robustness (Sampled)

- **Temperature:** 0.7 (sampling)
- **Seeds:** Multiple (default: [42, 123, 456])
- **Purpose:** Test robustness to sampling noise
- **Use case:** Distributional properties, variance estimation

### Standard Parameters

- **Max new tokens:** 100 (default)
- **Do sample:** True (Tier 2), False (Tier 1)
- **Pad token:** EOS token
- **Attention implementation:** "eager" (required for attention capture)

---

## Artifact Contract

### Standard Directory Structure

```
results/{experiment}/runs/{timestamp}_{name}/
├── config.json          # All parameters
├── summary.json         # Aggregated statistics
├── per_sample.csv      # Individual results
├── prompt_bank_version.json  # Hash of prompts/bank.json
└── {other artifacts}   # Experiment-specific
```

### Required Artifacts

1. **config.json:**
   - Model name
   - Measurement contract parameters
   - Prompt bank version
   - Seed, temperature, etc.

2. **summary.json:**
   - Aggregated statistics
   - Group-level means/stds
   - Separation statistics (Cohen's d)
   - Effect sizes

3. **per_sample.csv:**
   - Individual prompt results
   - All measured metrics
   - Prompt metadata (group, pillar, type)

4. **prompt_bank_version.json:**
   - Hash of prompts/bank.json
   - Timestamp of bank used
   - Enables exact reproducibility

---

## Validation Tests

### Test 1: Same Prompt → Same Number

- Run same prompt 10 times with Tier 1 (T=0, seed=42)
- All R_V values must be identical (within floating-point precision)
- **Pass criteria:** std(R_V) < 1e-6

### Test 2: Different Seeds → Different Numbers (Tier 2)

- Run same prompt with different seeds (T=0.7)
- R_V values should vary (sampling noise)
- **Pass criteria:** std(R_V) > 0.01 (shows sampling variance)

### Test 3: Prompt Bank Version Tracking

- Run audit with bank version A
- Modify bank (add prompt)
- Run audit with bank version B
- Versions must differ
- **Pass criteria:** version_A != version_B

---

## Edge Cases

### Short Prompts

- If prompt length < window_size (16 tokens):
  - Return NaN for R_V
  - Log warning
  - Continue processing other prompts

### Long Prompts

- If prompt length > 512 tokens:
  - Truncate to 512 (model limit)
  - Use last 512 tokens
  - Log truncation

### Degenerate Cases

- All-zero V-projections → NaN
- Constant V-projections → NaN
- Numerical instability in SVD → NaN

---

## Compliance

All experiments must:

1. ✅ Use `src/metrics/rv.py` for R_V computation
2. ✅ Document parameters in `config.json`
3. ✅ Record prompt bank version
4. ✅ Handle NaN cases consistently
5. ✅ Use Tier 1 (T=0) for primary measurements
6. ✅ Use Tier 2 (T=0.7) for robustness checks

---

## Version History

- **v1.1** (2024-12-16): Added Dynamic R_V definition for temporal stability (Pipeline 6).
- **v1.0** (2024-12-15): Initial contract definition

---

**This contract is LOCKED. Changes require explicit approval and version bump.**
