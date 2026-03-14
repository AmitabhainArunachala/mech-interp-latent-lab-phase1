# Multi-Token Bridge Experiment Results (v2 - GPT Audit Fixes)

**Date**: 2026-03-13 03:01
**Model**: mistralai/Mistral-7B-v0.1
**N prompts**: 72 total
**Recursive groups**: ['L5_refined', 'L4_full', 'L3_deeper']
**Baseline groups**: ['long_control', 'baseline_creative', 'baseline_math']
**Seed**: 42

## Temperature 0.0

**Truncation**: 62/72 (86.1%) truncated, 10 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count (non_truncated) | r=-0.285 | 4.25e-01 | No |
| H1a: all valid | r=-0.490 | 1.28e-05 | Yes |
| H1b: non-truncated | r=-0.285 | 4.25e-01 | No |
| H2: Recursive vs Baseline R_V | d=3.17 | 4.59e-21 | Yes |
| H3: L4 markers | r=-0.248 | 3.59e-02 | No |

**R_V means**: Recursive=0.502, Baseline=0.692

**Per-group R_V means**:
- L3_deeper: R_V=0.527
- L4_full: R_V=0.486
- L5_refined: R_V=0.492
- baseline_creative: R_V=0.674
- baseline_math: R_V=0.752
- long_control: R_V=0.651

## Temperature 0.7

**Truncation**: 51/72 (70.8%) truncated, 21 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count (non_truncated) | r=-0.688 | 5.62e-04 | Yes |
| H1a: all valid | r=-0.386 | 7.98e-04 | Yes |
| H1b: non-truncated | r=-0.688 | 5.62e-04 | Yes |
| H2: Recursive vs Baseline R_V | d=3.17 | 4.59e-21 | Yes |
| H3: L4 markers | r=-0.312 | 7.67e-03 | Yes |

**R_V means**: Recursive=0.502, Baseline=0.692

**Per-group R_V means**:
- L3_deeper: R_V=0.527
- L4_full: R_V=0.486
- L5_refined: R_V=0.492
- baseline_creative: R_V=0.674
- baseline_math: R_V=0.752
- long_control: R_V=0.651

## Overall Verdict

**PARTIAL CORRELATION - Investigate confounds**
