# Multi-Token Bridge Experiment Results (v2 - GPT Audit Fixes)

**Date**: 2026-01-24 16:58
**Model**: google/gemma-2-9b
**N prompts**: 117 total
**Recursive groups**: ['champions', 'L4_full', 'L3_deeper']
**Baseline groups**: ['baseline_factual', 'baseline_math', 'baseline_creative']
**Seed**: 42

## Temperature 0.0

**Truncation**: 99/117 (84.6%) truncated, 18 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count | r=-0.171 | 4.98e-01 | No |
| H2: Recursive vs Baseline R_V | d=3.37 | 1.10e-35 | Yes |
| H3: L4 markers | r=-0.241 | 8.84e-03 | Yes |

**R_V means**: Recursive=0.606, Baseline=0.777

**Per-group R_V means**:
- L3_deeper: R_V=0.607
- L4_full: R_V=0.592
- baseline_creative: R_V=0.771
- baseline_factual: R_V=0.795
- baseline_math: R_V=0.766
- champions: R_V=0.622

## Temperature 0.7

**Truncation**: 92/117 (78.6%) truncated, 25 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count | r=0.114 | 5.89e-01 | No |
| H2: Recursive vs Baseline R_V | d=3.37 | 1.10e-35 | Yes |
| H3: L4 markers | r=-0.178 | 5.53e-02 | No |

**R_V means**: Recursive=0.606, Baseline=0.777

**Per-group R_V means**:
- L3_deeper: R_V=0.607
- L4_full: R_V=0.592
- baseline_creative: R_V=0.771
- baseline_factual: R_V=0.795
- baseline_math: R_V=0.766
- champions: R_V=0.622

## Overall Verdict

**PARTIAL CORRELATION - Investigate confounds**
