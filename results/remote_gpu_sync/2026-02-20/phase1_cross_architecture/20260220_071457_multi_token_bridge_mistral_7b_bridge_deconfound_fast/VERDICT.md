# Multi-Token Bridge Experiment Results (v2 - GPT Audit Fixes)

**Date**: 2026-02-20 07:53
**Model**: mistralai/Mistral-7B-v0.1
**N prompts**: 36 total
**Recursive groups**: ['L5_refined', 'L4_full', 'L3_deeper']
**Baseline groups**: ['long_control', 'baseline_creative', 'baseline_math']
**Seed**: 42

## Temperature 0.0

**Truncation**: 32/36 (88.9%) truncated, 4 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count | r=-0.650 | 1.80e-05 | Yes |
| H2: Recursive vs Baseline R_V | d=3.54 | 2.52e-12 | Yes |
| H3: L4 markers | r=-0.167 | 3.31e-01 | No |

**R_V means**: Recursive=0.505, Baseline=0.700

**Per-group R_V means**:
- L3_deeper: R_V=0.524
- L4_full: R_V=0.497
- L5_refined: R_V=0.494
- baseline_creative: R_V=0.682
- baseline_math: R_V=0.742
- long_control: R_V=0.676

## Temperature 0.7

**Truncation**: 25/36 (69.4%) truncated, 11 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count | r=-0.409 | 2.12e-01 | No |
| H2: Recursive vs Baseline R_V | d=3.54 | 2.52e-12 | Yes |
| H3: L4 markers | r=nan | nan | No |

**R_V means**: Recursive=0.505, Baseline=0.700

**Per-group R_V means**:
- L3_deeper: R_V=0.524
- L4_full: R_V=0.497
- L5_refined: R_V=0.494
- baseline_creative: R_V=0.682
- baseline_math: R_V=0.742
- long_control: R_V=0.676

## Overall Verdict

**STRONG CORRELATION - Proceed to sufficiency tests**
