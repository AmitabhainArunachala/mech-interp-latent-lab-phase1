# L8 Experiments Running with Full Prompt Bank

**Started:** December 11, 2025  
**Status:** ✅ RUNNING IN BACKGROUND

---

## Experiment 1: Early-Layer Patching Sweep

**Status:** 🟢 RUNNING  
**Log File:** `l8_patching_sweep_350.log`

**Configuration:**
- **Layers:** [4, 8, 12, 16, 20, 24]
- **Prompt Pairs:** 100 (maximum available balanced pairs)
- **Total Iterations:** 100 pairs × 6 layers = 600 iterations
- **Estimated Time:** ~2 hours (13 seconds per iteration)

**What It's Testing:**
- Is L8 a discontinuity or part of a gradient?
- Does L8 patching = L8 steering?
- Where is the actual boundary?

**Output:** `results/dec11_evening/l8_early_layer_patching_sweep.csv`

---

## Experiment 2: L8 Ablation Test

**Status:** 🟢 RUNNING  
**Log File:** `l8_ablation_350.log`

**Configuration:**
- **Recursive Prompts:** 105 (all available from L1-L5)
  - L1_hint: 20
  - L2_simple: 20
  - L3_deeper: 25
  - L4_full: 20
  - L5_refined: 20
- **Baseline Prompts:** 100 (all available)
  - baseline_math: 20
  - baseline_factual: 20
  - baseline_creative: 20
  - baseline_impossible: 20
  - baseline_personal: 20
- **Total:** 205 prompts
- **Estimated Time:** ~1 hour (1.8 seconds per prompt)

**What It's Testing:**
- Does L8 ablation break recursive prompts differently than baseline?
- Is L8 necessary for R_V contraction?
- What happens to outputs when L8 is ablated?

**Output:** `results/dec11_evening/l8_ablation_test.csv`

---

## Monitoring Progress

### Check Experiment 1 Status:
```bash
tail -f l8_patching_sweep_350.log
```

### Check Experiment 2 Status:
```bash
tail -f l8_ablation_350.log
```

### Check if processes are running:
```bash
ps aux | grep experiment_l8
```

### Check progress by counting completed pairs:
```bash
# Experiment 1
grep "Pair.*L24" l8_patching_sweep_350.log | wc -l

# Experiment 2
grep "Recursive.*Normal:" l8_ablation_350.log | wc -l
grep "Baseline.*Normal:" l8_ablation_350.log | wc -l
```

---

## Expected Results

### Experiment 1 (100 pairs):
- **Much larger sample size** than initial run (10 pairs)
- **More statistical power** to detect layer differences
- **Better characterization** of the L4→L8→L24 gradient
- **More robust** state transition patterns

### Experiment 2 (205 prompts):
- **Comprehensive coverage** of all prompt types
- **Dose-response analysis** across L1-L5 recursive levels
- **Baseline diversity** across all baseline groups
- **Stronger evidence** for L8 necessity/importance

---

## Comparison to Initial Run

| Metric | Initial Run | Full Run |
|--------|-------------|----------|
| **Experiment 1 Pairs** | 10 | 100 (10x) |
| **Experiment 2 Prompts** | 20 (10+10) | 205 (10x) |
| **Total Data Points** | 60 + 20 = 80 | 600 + 205 = 805 |
| **Statistical Power** | Low | High |
| **Runtime** | ~35 min | ~3 hours |

---

## Next Steps After Completion

1. **Analyze results** with larger sample size
2. **Compare** to initial 10-pair run
3. **Validate** findings with statistical tests
4. **Update** L8_GAP_FILLING_RESULTS.md with full findings
5. **Determine** if L8 is truly part of a gradient or has special properties

---

## Files Generated

- `l8_patching_sweep_350.log` - Experiment 1 log
- `l8_ablation_350.log` - Experiment 2 log
- `results/dec11_evening/l8_early_layer_patching_sweep.csv` - Experiment 1 results
- `results/dec11_evening/l8_ablation_test.csv` - Experiment 2 results

---

**Both experiments are running in the background. Check logs for progress!**
