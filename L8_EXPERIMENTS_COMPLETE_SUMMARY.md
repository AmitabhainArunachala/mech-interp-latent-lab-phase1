# ✅ L8 Gap-Filling Experiments - COMPLETE

**Date:** December 11, 2025  
**Status:** ✅ **BOTH EXPERIMENTS COMPLETE - SAFE TO SHUTDOWN**

---

## 📊 Experiment Status

### ✅ Experiment 1: Early-Layer Bidirectional Patching Sweep
- **Status:** COMPLETE
- **File:** `results/dec11_evening/l8_early_layer_patching_sweep.csv`
- **Data:** 600 rows (100 pairs × 6 layers)
- **Layers tested:** [4, 8, 12, 16, 20, 24]
- **Completion time:** ~28 minutes

### ✅ Experiment 2: L8 Ablation Test
- **Status:** COMPLETE
- **File:** `results/dec11_evening/l8_ablation_test.csv`
- **Data:** 205 prompts (105 recursive + 100 baseline)
- **Completion time:** ~15 minutes

---

## 🔍 Key Findings from Experiment 1

### Layer-by-Layer Effects (rec→base patching):

| Layer | Mean RV Δ | Collapse Rate | Interpretation |
|-------|-----------|---------------|----------------|
| **L4** | -0.017 | 31% | Minimal effect |
| **L8** | -0.332 | 72% | **Strong effect** |
| **L12** | -0.458 | 79% | **Strongest effect** |
| **L16** | -0.397 | 98% | Very strong, near-universal collapse |
| **L20** | -0.381 | 99% | Near-universal collapse |
| **L24** | -0.335 | 99% | Near-universal collapse |

### Critical Discovery:

**L8 is NOT a discontinuity** — it's part of a **smooth gradient**:
- L4: Minimal effect (31% collapse)
- L8: Moderate effect (72% collapse) 
- L12: Peak effect (79% collapse, largest RV delta)
- L16-24: Near-universal collapse (98-99%)

**L8 is NOT unique** — L12 shows the strongest effect, and L16+ show near-universal collapse.

---

## 🔍 Key Findings from Experiment 2

### L8 Ablation Effects:

- **Recursive prompts:** 33.3% state change (35/105)
- **Baseline prompts:** 22.0% state change (22/100)
- **RV delta (recursive):** +0.0017 (minimal)
- **RV delta (baseline):** +0.0041 (minimal)

### Statistical Tests:
- **State change rate difference:** p=0.0980 (not significant)
- **RV delta difference:** p=0.1647 (not significant)

### Interpretation:
- L8 ablation affects recursive prompts more than baseline (33% vs 22%)
- But the effect is **smaller than expected** and **not statistically significant**
- L8 is important for recursive behavior but **not the sole source** of R_V contraction

---

## 🎯 Answers to Original Questions

### Q1: Is L8 a discontinuity or part of a smooth gradient?
**A:** **Smooth gradient** — L8 is part of a progression from L4 (minimal) → L8 (moderate) → L12 (peak) → L16+ (universal)

### Q2: Is L8 patching equivalent to L8 steering?
**A:** **No** — Patching at L8 causes 72% collapse, while steering causes "interrogative mode" → collapse. Different mechanisms.

### Q3: Does L8 ablation affect recursive prompts differently?
**A:** **Slightly** — 33% vs 22% state change, but **not statistically significant**. Effect is smaller than expected.

### Q4: Where is the actual boundary layer for patching effects?
**A:** **L12** shows the strongest effect (largest RV delta: -0.458). L16+ show near-universal collapse (98-99%).

---

## 📁 Files Saved

### Results CSVs:
1. ✅ `results/dec11_evening/l8_early_layer_patching_sweep.csv` (569 KB, 600 rows)
2. ✅ `results/dec11_evening/l8_ablation_test.csv` (137 KB, 205 rows)

### Log Files:
1. ✅ `l8_patching_sweep_350.log` (complete experiment log)
2. ✅ `l8_ablation_350.log` (complete experiment log)

---

## 🚨 Shutdown Status

**✅ SAFE TO SHUTDOWN**

Both experiments are complete, all data is saved, and results are verified.

---

## 📝 Next Steps (Future Work)

1. **Statistical analysis:** Full chi-square and t-tests comparing layers
2. **L12 investigation:** Why does L12 show the strongest effect?
3. **Steering vs. patching:** Deeper comparison of mechanisms
4. **Cross-model validation:** Test if L8/L12 pattern holds in other models

---

**Experiments completed successfully!** 🎉
