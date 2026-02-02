# Stage 2: Canonical Suite - Progress Report

**Date:** January 5, 2025  
**Status:** 🟡 In Progress

---

## Overview

Running 7 core canonical experiments with standardized infrastructure:
- ✅ Prompt IDs tracked
- ✅ Standardized metadata
- ✅ RUN_INDEX.jsonl tracking
- ✅ n_pairs=30 for all tests

---

## Experiments Status

### 1. ✅ L0 Necessity Test
- **Status:** ✅ Completed
- **Run Directory:** `results/canonical_suite_v1_0/runs/20260105_140742_l0_necessity`
- **Result:** 
  - R_V baseline: 0.7124 ± 0.1511 (contracted, < 1.0)
  - R_V ablated: 1.5225 ± 0.2861 (expanded, > 1.0)
  - R_V delta: +0.8101 ± 0.3188 (p < 10⁻¹⁴, Cohen's d = 2.58)
  - **Interpretation:** L0 ablation REMOVES contraction → **L0 IS NECESSARY**
  - **Note:** Verdict message in code is inverted, but data is clear
  - **Prompt IDs:** Tracked (30 recursive + 30 baseline)
  - **Metadata:** ✅ Git commit, prompt bank version, model ID all logged

### 2. 🟡 L1 Necessity Test
- **Status:** 🟡 Running
- **Log:** `/tmp/canonical_l1_necessity.log`
- **Started:** ~14:14 UTC

### 3. 🟡 L2 Necessity Test
- **Status:** 🟡 Running
- **Log:** `/tmp/canonical_l2_necessity.log`
- **Started:** ~14:14 UTC

### 4. 🟡 L3 Necessity Test
- **Status:** 🟡 Running
- **Log:** `/tmp/canonical_l3_necessity.log`
- **Started:** ~14:14 UTC

### 5. 🟡 L0 Sufficiency Test
- **Status:** 🟡 Running
- **Log:** `/tmp/canonical_l0_sufficiency.log`
- **Started:** ~14:14 UTC

### 6. 🟡 L0+L1 Combined Sufficiency Test
- **Status:** 🟡 Running
- **Log:** `/tmp/canonical_l0_l1_sufficiency.log`
- **Started:** ~14:14 UTC

### 7. 🟡 L0 Position-Specific Test
- **Status:** 🟡 Running
- **Log:** `/tmp/canonical_l0_position.log`
- **Started:** ~14:14 UTC

---

## Monitoring Commands

```bash
# Check all logs
ssh runpod-current "tail -f /tmp/canonical_*.log"

# Check specific experiment
ssh runpod-current "tail -f /tmp/canonical_l1_necessity.log"

# Check run directories
ssh runpod-current "ls -lh results/canonical_suite_v1_0/runs/"

# Check RUN_INDEX
ssh runpod-current "tail -20 results/RUN_INDEX.jsonl"
```

---

## Expected Completion Time

- Each experiment: ~5-6 minutes (30 pairs × ~10s per pair)
- Total suite: ~35-40 minutes (parallel execution)

---

## Next Steps

1. ✅ Monitor all 6 running experiments
2. ⏳ Wait for completion
3. ⏳ Collect results and create summary
4. ⏳ Compare with previous runs (if available)
5. ⏳ Create Stage 2 Final Report

---

**Last Updated:** January 5, 2025 14:15 UTC

