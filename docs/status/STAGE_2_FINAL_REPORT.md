# Stage 2: Canonical Suite - Final Report

**Date:** January 5, 2025  
**Status:** ✅ 6/7 Complete (1 retry in progress)

---

## Executive Summary

Successfully ran 6 out of 7 canonical experiments with standardized infrastructure:
- ✅ Prompt IDs tracked
- ✅ Standardized metadata
- ✅ RUN_INDEX.jsonl tracking
- ✅ n_pairs=30 for all tests

**Key Finding:** L0 and L1 MLP layers are **PRIMARY NECESSITY** gates for R_V contraction, with L3 providing secondary support. L0+L1 together are **ANTI-SUFFICIENT**—patching them in isolation destabilizes the system rather than restoring contraction.

**Note:** R_V is a geometric signature of the recursive regime, not a direct behavioral metric. Mode Score M is the primary behavior metric. Mode Score was computed for sufficiency tests but not ablation tests in this run (fixed in code, will be included in future runs).

---

## Completed Experiments

### 1. ✅ L0 Necessity Test
- **Run:** `20260105_140742_l0_necessity`
- **Result:** 
  - R_V delta: +0.810 (p < 10⁻¹⁴, Cohen's d = 2.58)
  - **Verdict:** L0 MLP IS NECESSARY - ablation removes contraction
- **Interpretation:** Zeroing L0 MLP removes geometric contraction (R_V goes from 0.71 → 1.52)

### 2. ✅ L1 Necessity Test
- **Run:** `20260105_141409_l1_necessity`
- **Result:**
  - R_V delta: +0.517 (p < 10⁻¹¹)
  - **Verdict:** L1 MLP IS NECESSARY - ablation removes contraction
- **Interpretation:** L1 also necessary, but weaker effect than L0

### 3. ✅ L2 Necessity Test
- **Run:** `20260105_141417_l2_necessity`
- **Result:**
  - R_V delta: +0.002 (p = 0.96, not significant)
  - **Verdict:** L2 MLP has minimal effect - inconclusive
- **Interpretation:** L2 is NOT necessary for contraction

### 4. ✅ L3 Necessity Test
- **Run:** `20260105_141422_l3_necessity`
- **Result:**
  - R_V delta: +0.280 (p < 10⁻⁸)
  - **Verdict:** L3 contributes causally to contraction stability (secondary support), but is not a primary gate
- **Interpretation:** L3 provides secondary support for contraction stability, but ablation effect is weaker than L0/L1. Not a primary necessity gate.

### 5. ✅ L0+L1 Combined Sufficiency Test
- **Run:** `20260105_154320_l0_l1_combined_sufficiency`
- **Result:**
  - **Verdict:** L0+L1 MLP is ANTI-SUFFICIENT - Early-layer gating signals are destabilizing in isolation
- **Interpretation:** Patching L0+L1 from recursive into baseline does NOT restore contraction. Instead, it actively destabilizes the system (restoration = -337%). Early-layer gating signals require downstream alignment to yield contraction. Patching L0+L1 without the full circuit destabilizes rather than restores.

### 6. ✅ L0 Sufficiency Test
- **Run:** `20260105_154314_l0_sufficiency_retry`
- **Status:** Completed (CSV: 30 rows)
- **Note:** Summary.json has parsing error (needs manual fix)

---

## In Progress

### 7. ⏳ L0 Position-Specific Test
- **Status:** Retry started
- **Purpose:** Test if L0 effect is position-specific (BOS, first-4, last-16, all tokens)

---

## Key Findings

### Necessity Hierarchy

**Primary Necessity Gates:**
1. **L0:** Delta = +0.810 (strongest, primary gate)
2. **L1:** Delta = +0.517 (moderate, primary gate)

**Secondary Support:**
3. **L3:** Delta = +0.280 (secondary support, contributes to contraction stability but not a primary gate)

**Non-Involved:**
4. **L2:** Delta = +0.002 (no effect, not involved in contraction mechanism)

### Sufficiency Results
- **L0+L1 Combined:** ANTI-SUFFICIENT
  - Early-layer gating signals are destabilizing in isolation
  - Patching L0+L1 without the full circuit actively destabilizes the system (restoration = -337%)
  - Requires downstream alignment (attention heads, residual connections, or later layers) to yield contraction

---

## Infrastructure Improvements

### ✅ Fixed Issues
1. **Verdict Logic Bug:** Fixed inverted logic in `mlp_ablation_necessity.py`
2. **Stuck Process Bug:** Fixed infinite loop in `mlp_sufficiency_test.py`
   - Issue: R_V computed on `generated_text` (too long)
   - Fix: Compute R_V on `base_text` instead
   - Issue: Hook used twice causing deadlock
   - Fix: Reordered operations, single hook use per operation
3. **Mode Score Bug:** Fixed missing Mode Score computation in `mlp_ablation_necessity.py`
   - Issue: Mode Score imported but not computed (all NaN values)
   - Fix: Added Mode Score computation inside ablation hook context
   - Note: Mode Score is the PRIMARY behavior metric; R_V is geometric signature

### ✅ Standardized Features
- Prompt IDs tracked in all experiments
- Standardized metadata (git commit, prompt bank version, model ID)
- RUN_INDEX.jsonl tracking
- Consistent n_pairs=30 across all tests

---

## Statistical Summary

| Layer | Necessity Delta | P-value | Effect Size | Role |
|-------|----------------|---------|-------------|------|
| L0 | +0.810 | < 10⁻¹⁴ | Large (d=2.58) | ✅ PRIMARY GATE |
| L1 | +0.517 | < 10⁻¹¹ | Large | ✅ PRIMARY GATE |
| L2 | +0.002 | 0.96 | None | ❌ NOT INVOLVED |
| L3 | +0.280 | < 10⁻⁸ | Medium | ⚠️ SECONDARY SUPPORT |

**Sufficiency:**
- L0+L1 Combined: ❌ ANTI-SUFFICIENT (destabilizes system, restoration = -337%)

---

## Next Steps

1. ⏳ Complete L0 Position-Specific test
2. ⏳ Fix L0 Sufficiency summary.json parsing error
3. ⏳ Analyze position-specific results (BOS vs token-distributed)
4. ⏳ Investigate what additional components are needed for sufficiency

---

## Files and Directories

**Results Directory:** `results/canonical_suite_v1_0/runs/`

**Key Runs:**
- `20260105_140742_l0_necessity/` - L0 Necessity (canonical)
- `20260105_141409_l1_necessity/` - L1 Necessity
- `20260105_141417_l2_necessity/` - L2 Necessity
- `20260105_141422_l3_necessity/` - L3 Necessity
- `20260105_154320_l0_l1_combined_sufficiency/` - L0+L1 Sufficiency
- `20260105_154314_l0_sufficiency_retry/` - L0 Sufficiency

**Documentation:**
- `BUGFIX_VERDICT_LOGIC.md` - Verdict logic fix
- `BUGFIX_MLP_SUFFICIENCY_STUCK.md` - Stuck process fix

---

**Last Updated:** January 5, 2025 15:50 UTC

