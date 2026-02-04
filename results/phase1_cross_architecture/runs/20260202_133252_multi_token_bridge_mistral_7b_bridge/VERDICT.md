# Multi-Token Bridge Experiment Results (v2 - GPT Audit Fixes)

**Date**: 2026-02-02 13:44
**Model**: mistralai/Mistral-7B-v0.1
**N prompts**: 120 total
**Recursive groups**: ['L5_refined', 'L4_full', 'L3_deeper']
**Baseline groups**: ['long_control', 'baseline_creative', 'baseline_math']
**Seed**: 42

## Temperature 0.0

**Truncation**: 111/120 (92.5%) truncated, 9 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count | r=-0.183 | 6.37e-01 | No |
| H2: Recursive vs Baseline R_V | d=2.90 | 4.38e-31 | Yes |
| H3: L4 markers | r=-0.230 | 1.16e-02 | No |

**R_V means**: Recursive=0.506, Baseline=0.687

**Per-group R_V means**:
- L3_deeper: R_V=0.523
- L4_full: R_V=0.497
- L5_refined: R_V=0.497
- baseline_creative: R_V=0.651
- baseline_math: R_V=0.741
- long_control: R_V=0.669

## Temperature 0.7

**Truncation**: 104/120 (86.7%) truncated, 16 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count | r=-0.761 | 6.20e-04 | Yes |
| H2: Recursive vs Baseline R_V | d=2.90 | 4.38e-31 | Yes |
| H3: L4 markers | r=-0.286 | 1.52e-03 | Yes |

**R_V means**: Recursive=0.506, Baseline=0.687

**Per-group R_V means**:
- L3_deeper: R_V=0.523
- L4_full: R_V=0.497
- L5_refined: R_V=0.497
- baseline_creative: R_V=0.651
- baseline_math: R_V=0.741
- long_control: R_V=0.669

## Overall Verdict

**PARTIAL CORRELATION - Investigate confounds**
