# MLP Ablation Necessity Test

**Date:** January 5, 2025  
**Status:** Ready to run

---

## Goal

Test if **L0 MLP is NECESSARY** for recursive behavior by zeroing it out.

---

## Hypothesis

**If L0 MLP is NECESSARY:**
- R_V contraction **disappears** (R_V → 1.0, delta > +0.1)
- Recursive behavior **stops** (mode score → baseline)
- Generated text loses recursive content

**If L0 MLP is NOT necessary:**
- R_V contraction **persists** (delta ≈ 0)
- Recursive behavior **continues** (mode score unchanged)
- Generated text maintains recursive content

---

## Method

1. **Baseline:** Generate text from recursive prompts (no ablation)
2. **Ablation:** Zero out L0 MLP output, generate same prompts
3. **Measure:**
   - R_V delta (geometry change)
   - Mode score delta (behavior change)
   - Coherence & recursion scores (output quality)

---

## Experimental Design

- **Model:** Mistral-7B-v0.1 (BASE)
- **Layer:** L0 MLP
- **Pairs:** 80 (protocol minimum)
- **Generation:** Deterministic (temperature=0.0)
- **Metrics:** R_V, Mode Score, Coherence, Recursion Score

---

## Expected Results

### Scenario A: L0 MLP is NECESSARY
```
R_V baseline:  0.85 (contraction)
R_V ablated:   1.00 (no contraction)
R_V delta:     +0.15 (SIGNIFICANT, p < 0.01)

Mode baseline:  +0.5 (recursive)
Mode ablated:   0.0 (baseline)
Mode delta:     -0.5 (SIGNIFICANT)
```

### Scenario B: L0 MLP is NOT necessary
```
R_V baseline:  0.85 (contraction)
R_V ablated:   0.85 (contraction persists)
R_V delta:     ~0.0 (NOT significant)

Mode baseline:  +0.5 (recursive)
Mode ablated:   +0.5 (recursive persists)
Mode delta:     ~0.0 (NOT significant)
```

---

## Statistical Testing

- **One-sample t-test:** Is delta significantly different from zero?
- **Threshold:** p < 0.01 (Bonferroni correction)
- **Effect size:** Cohen's d ≥ 0.5 for meaningful effect

---

## Files

- **Pipeline:** `src/pipelines/mlp_ablation_necessity.py`
- **Config:** `configs/mlp_ablation_necessity_l0.json`
- **Results:** `results/phase1_mechanism/runs/<timestamp>_l0_necessity_test/`

---

## Run Command

```bash
python -m src.pipelines.run --config configs/mlp_ablation_necessity_l0.json
```

---

## Interpretation

**Verdict Logic:**
- If `rv_delta > +0.1` and `p < 0.01`: **L0 MLP is NECESSARY**
- If `rv_delta ≈ 0` and `p > 0.01`: **L0 MLP is NOT necessary**
- Otherwise: **Inconclusive**

---

## Next Steps

1. Run L0 ablation test
2. If L0 is necessary → Test L1, L2, L3 (find minimal necessary set)
3. If L0 is not necessary → Test later layers (L3-L5 where steering worked)


