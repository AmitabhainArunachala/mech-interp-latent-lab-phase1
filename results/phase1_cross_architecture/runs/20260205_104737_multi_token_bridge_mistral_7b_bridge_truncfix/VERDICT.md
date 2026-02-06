# Multi-Token Bridge Experiment Results (v2 - GPT Audit Fixes)

**Date**: 2026-02-05 11:10
**Model**: mistralai/Mistral-7B-v0.1
**N prompts**: 120 total
**Recursive groups**: ['L5_refined', 'L4_full', 'L3_deeper']
**Baseline groups**: ['long_control', 'baseline_creative', 'baseline_math']
**Seed**: 42

## Temperature 0.0

**Truncation**: 108/120 (90.0%) truncated, 12 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count | r=-0.543 | 6.82e-02 | No |
| H2: Recursive vs Baseline R_V | d=2.95 | 1.01e-31 | Yes |
| H3: L4 markers | r=-0.246 | 6.81e-03 | Yes |

**R_V means**: Recursive=0.505, Baseline=0.685

**Per-group R_V means**:
- L3_deeper: R_V=0.523
- L4_full: R_V=0.496
- L5_refined: R_V=0.494
- baseline_creative: R_V=0.651
- baseline_math: R_V=0.734
- long_control: R_V=0.669

## Temperature 0.7

**Truncation**: 99/120 (82.5%) truncated, 21 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count | r=-0.336 | 1.37e-01 | No |
| H2: Recursive vs Baseline R_V | d=2.95 | 1.01e-31 | Yes |
| H3: L4 markers | r=-0.204 | 2.57e-02 | No |

**R_V means**: Recursive=0.505, Baseline=0.685

**Per-group R_V means**:
- L3_deeper: R_V=0.523
- L4_full: R_V=0.496
- L5_refined: R_V=0.494
- baseline_creative: R_V=0.651
- baseline_math: R_V=0.734
- long_control: R_V=0.669

## Overall Verdict

**PARTIAL CORRELATION - Investigate confounds**
