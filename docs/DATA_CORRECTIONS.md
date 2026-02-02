# Data Corrections Log
**Date:** 2026-01-25
**Status:** ACTIVE

---

## Summary

Following a 6-agent review (3 Claude + 3 GPT), several data discrepancies were identified. This document tracks corrections.

---

## Corrections Applied

### 1. EOS Baseline Rate: 45% → 30%

| Metric | Claimed | Actual | Source |
|--------|---------|--------|--------|
| Baseline EOS rate | 45% | **30%** (18/60) | summary.json, lines 32-34 |

**Files corrected:**
- `docs/sessions/2026-01-24_gemma_multi_token_bridge_v3.md` ✓
- `docs/sessions/2026-01-25_STRATEGIC_BRAINSTORM.md` ✓
- `docs/sessions/2026-01-25_TODO_IDEAS.md` ✓

**Files still containing 45% (historical - will correct as touched):**
- Various older session docs (now superseded by audit)

---

### 2. Cohen's d (Mistral): -3.56 → -3.558

| Metric | Claimed | Actual | Source |
|--------|---------|--------|--------|
| Cohen's d | -3.56 | **-3.558** | MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md, line 46 |

**Note:** -3.56 is a valid rounding of -3.558. Both represent the same measurement. For consistency, use -3.558 (precise) in code/data, -3.56 (rounded) is acceptable in prose.

**Files corrected:**
- `~/CLAUDE.md` ✓
- `docs/sessions/2026-01-25_TODO_IDEAS.md` ✓

---

### 3. PR Formula Clarification

**Code implementation:**
```python
PR = (Σλᵢ²)² / Σ(λᵢ⁴)  # Squared singular values
```

**Some docs say:**
```
PR = (Σλᵢ)² / Σλᵢ²  # Raw singular values (eigenvalues)
```

**Resolution:** Both are valid PR formulations. The code uses the squared-singular-value version, which is standard for measuring effective rank. Documents should specify which formulation is used.

---

## Files Flagged for Future Correction

These files contain historical values that may be updated as they're touched:

| File | Issue | Priority |
|------|-------|----------|
| `docs/misc/VERIFIED_SIGNALS.md` | -3.56 | Low |
| `docs/misc/THE_CONNECTION_ESSAY.md` | -3.56 | Low |
| `docs/analysis/AUDIT.md` | -3.56 | Medium |
| `REPOSITORY_DISSECTION_COMPLETE.md` | -3.56 | Medium |
| `R_V_PAPER/README.md` | -3.56 | High (before publication) |
| `visualizations/*/main.py` | Hardcoded -3.56 | Low (cosmetic) |

---

## Verified Correct Values

For reference, these are the authoritative values from source data:

### Gemma 2 9B (Multi-Token Bridge V3)
- **Cohen's d (H2):** 3.369 (baseline vs recursive R_V)
- **EOS reached:** 18/117 (15.4% overall, 30% of baseline)
- **R_V recursive:** 0.606
- **R_V baseline:** 0.777

### Mistral-7B (Causal Validation)
- **Cohen's d:** -3.558
- **n:** 45 pairs
- **Transfer efficiency:** 117.8%
- **p-value:** < 10⁻⁶

---

## Audit Reference

Full details in: `docs/sessions/2026-01-25_CONSOLIDATED_AUDIT_ACTION_PLAN.md`

---

*Last updated: 2026-01-25 by Claude Code*
