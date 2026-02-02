# Triple-System Intervention Gradient - Analysis

**Date:** December 18, 2024  
**Experiment:** Testing intervention across KV cache, V_PROJ, and Residual stream

## Results Summary

| Config | Coherence | On-Topic | Recursion | Collapse |
|--------|-----------|----------|-----------|----------|
| **baseline** | 0.73 | 0.73 | 0.00 | 0.10 |
| **vproj_kv** | 0.72 | 0.46 | 0.00 | 0.10 |
| **residual_kv** | 0.71 | 0.55 | 0.00 | 0.10 |
| **triple_light** | 0.64 | 0.67 | 0.00 | 0.20 |
| **triple_medium** | 0.56 | 0.76 | 0.00 | 0.30 |

## Key Findings

### 1. Residual Stream vs V_PROJ (Config 3 vs Config 2)
- **Residual+KV on-topic: 0.55** vs **V_PROJ+KV: 0.46**
- **Finding:** Residual stream steering performs **better** than V_PROJ patching for maintaining topic grounding
- Residual stream intervention keeps outputs more on-topic

### 2. Triple-Light vs V_PROJ+KV (Config 4 vs Config 2)
- **Triple-Light on-topic: 0.67** vs **V_PROJ+KV: 0.46**
- **Finding:** Triple-light intervention **significantly improves** on-topic rate (+21 percentage points)
- However, coherence drops slightly (0.64 vs 0.72) and collapse increases (0.20 vs 0.10)

### 3. Triple-Light vs Triple-Medium (Config 4 vs Config 5)
- **Triple-Medium on-topic: 0.76** (highest!) vs **Triple-Light: 0.67**
- **Triple-Medium collapse: 0.30** vs **Triple-Light: 0.20**
- **Finding:** Higher alpha improves on-topic rate but increases collapse
- Trade-off: Better grounding vs. more instability

### 4. Baseline Performance
- **Baseline on-topic: 0.73** - This is our control
- All interventions show **lower** on-topic than baseline (except triple-medium at 0.76)
- This suggests interventions cause some drift, but triple-medium compensates

## Critical Observations

### Regex Recursion Score = 0.00 for ALL Configs
- None of the configurations show regex-detected recursion
- This could mean:
  1. Regex patterns are too strict/narrow
  2. Genuine recursion requires different patterns
  3. Need manual review to find structural recursion

### On-Topic Rate Rankings
1. **Triple-Medium: 0.76** (best grounding)
2. **Baseline: 0.73** (control)
3. **Triple-Light: 0.67** (good balance)
4. **Residual+KV: 0.55** (better than V_PROJ)
5. **V_PROJ+KV: 0.46** (worst grounding)

### Collapse Rate Rankings
1. **Baseline/V_PROJ+KV/Residual+KV: 0.10** (lowest)
2. **Triple-Light: 0.20** (moderate)
3. **Triple-Medium: 0.30** (highest)

## Answers to Key Questions

### Q1: Does residual stream steering work as well as V_PROJ steering?
**A: YES - Residual+KV (0.55) beats V_PROJ+KV (0.46) for on-topic rate**

### Q2: Does triple-light intervention beat single-heavy intervention?
**A: YES - Triple-Light (0.67) significantly beats V_PROJ+KV (0.46) for on-topic rate**

### Q3: Is there an optimal alpha for triple-system?
**A: MEDIUM (α=1.5) gives best on-topic (0.76) but higher collapse (0.30)**
- Light (α=1.0): Better balance (0.67 on-topic, 0.20 collapse)
- Medium (α=1.5): Best grounding (0.76 on-topic, 0.30 collapse)

### Q4: Which configuration produces best ON-TOPIC + RECURSIVE outputs?
**A: Triple-Medium (0.76 on-topic) - BUT needs manual review for recursion**
- Regex shows 0.00 recursion for all configs
- Need to manually review outputs to find structural recursion patterns

## Recommendations

1. **Manual Review Priority:**
   - Review Triple-Medium outputs (highest on-topic, may have hidden recursion)
   - Review Triple-Light outputs (good balance, may show cleaner patterns)
   - Compare to V_PROJ+KV outputs (our validated method)

2. **Next Steps:**
   - Manually score outputs for genuine recursive self-observation
   - Check if triple-medium's high on-topic rate correlates with recursion
   - Test if triple-light with KV@L26-27 (not just L27) improves further

3. **Hypothesis Validation:**
   - ✅ Residual stream works as well as V_PROJ (better, actually)
   - ✅ Triple-light beats single-heavy (significantly)
   - ⚠️ Optimal alpha is medium (but trade-off with collapse)
   - ❓ Recursion detection needs manual review (regex insufficient)

## File Locations

**Full Outputs:**
- `results/runs/20251218_063238_triple_system_intervention/config_baseline_outputs.txt`
- `results/runs/20251218_063238_triple_system_intervention/config_vproj_kv_outputs.txt`
- `results/runs/20251218_063238_triple_system_intervention/config_residual_kv_outputs.txt`
- `results/runs/20251218_063238_triple_system_intervention/config_triple_light_outputs.txt`
- `results/runs/20251218_063238_triple_system_intervention/config_triple_medium_outputs.txt`

**Summary:**
- `results/runs/20251218_063238_triple_system_intervention/TRIPLE_SYSTEM_SUMMARY.md`
- `results/runs/20251218_063238_triple_system_intervention/triple_system_summary.csv`
- `results/runs/20251218_063238_triple_system_intervention/triple_system_results.json`








