# Multi-Token Bridge Experiment Results (v2 - GPT Audit Fixes)

**Date**: 2026-03-13 06:25
**Model**: mistralai/Mistral-7B-v0.1
**N prompts**: 122 total
**Recursive groups**: ['L5_refined', 'L4_full', 'L3_deeper']
**Baseline groups**: ['long_control', 'baseline_creative', 'baseline_math']
**Seed**: 42

## Temperature 0.7

**Truncation**: 86/122 (70.5%) truncated, 36 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count (non_truncated) | r=-0.070 | 6.86e-01 | No |
| H1a: all valid | r=-0.366 | 3.30e-05 | Yes |
| H1b: non-truncated | r=-0.070 | 6.86e-01 | No |
| H2: Recursive vs Baseline R_V | d=2.91 | 1.19e-31 | Yes |
| H3: L4 markers | r=-0.267 | 2.98e-03 | Yes |

**R_V means**: Recursive=0.506, Baseline=0.687

**Per-group R_V means**:
- L3_deeper: R_V=0.523
- L4_full: R_V=0.497
- L5_refined: R_V=0.497
- baseline_creative: R_V=0.651
- baseline_math: R_V=0.741
- long_control: R_V=0.669

