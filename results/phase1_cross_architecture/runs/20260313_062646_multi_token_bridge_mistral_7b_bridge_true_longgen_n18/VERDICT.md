# Multi-Token Bridge Experiment Results (v2 - GPT Audit Fixes)

**Date**: 2026-03-13 07:08
**Model**: mistralai/Mistral-7B-v0.1
**N prompts**: 108 total
**Recursive groups**: ['L5_refined', 'L4_full', 'L3_deeper']
**Baseline groups**: ['long_control', 'baseline_creative', 'baseline_math']
**Seed**: 42

## Temperature 0.7

**Truncation**: 67/108 (62.0%) truncated, 41 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count (non_truncated) | r=0.036 | 8.22e-01 | No |
| H1a: all valid | r=-0.339 | 3.38e-04 | Yes |
| H1b: non-truncated | r=0.036 | 8.22e-01 | No |
| H2: Recursive vs Baseline R_V | d=2.97 | 6.39e-29 | Yes |
| H3: L4 markers | r=-0.231 | 1.60e-02 | No |

**R_V means**: Recursive=0.505, Baseline=0.687

**Per-group R_V means**:
- L3_deeper: R_V=0.521
- L4_full: R_V=0.492
- L5_refined: R_V=0.502
- baseline_creative: R_V=0.652
- baseline_math: R_V=0.742
- long_control: R_V=0.665

