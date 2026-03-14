# Multi-Token Bridge Experiment Results (v2 - GPT Audit Fixes)

**Date**: 2026-03-13 02:22
**Model**: mistralai/Mistral-7B-v0.1
**N prompts**: 120 total
**Recursive groups**: ['L5_refined', 'L4_full', 'L3_deeper']
**Baseline groups**: ['baseline_factual', 'baseline_creative', 'baseline_math']
**Seed**: 42

## Temperature 0.0

**Truncation**: 111/120 (92.5%) truncated, 9 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count (non_truncated) | r=-0.450 | 2.24e-01 | No |
| H1a: all valid | r=-0.423 | 1.47e-06 | Yes |
| H1b: non-truncated | r=-0.450 | 2.24e-01 | No |
| H2: Recursive vs Baseline R_V | d=3.15 | 4.20e-34 | Yes |
| H3: L4 markers | r=-0.271 | 2.78e-03 | Yes |

**R_V means**: Recursive=0.506, Baseline=0.707

**Per-group R_V means**:
- L3_deeper: R_V=0.523
- L4_full: R_V=0.497
- L5_refined: R_V=0.497
- baseline_creative: R_V=0.651
- baseline_factual: R_V=0.729
- baseline_math: R_V=0.741

## Temperature 0.7

**Truncation**: 95/120 (79.2%) truncated, 25 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count (non_truncated) | r=-0.242 | 2.44e-01 | No |
| H1a: all valid | r=-0.267 | 3.24e-03 | Yes |
| H1b: non-truncated | r=-0.242 | 2.44e-01 | No |
| H2: Recursive vs Baseline R_V | d=3.15 | 4.20e-34 | Yes |
| H3: L4 markers | r=-0.321 | 3.47e-04 | Yes |

**R_V means**: Recursive=0.506, Baseline=0.707

**Per-group R_V means**:
- L3_deeper: R_V=0.523
- L4_full: R_V=0.497
- L5_refined: R_V=0.497
- baseline_creative: R_V=0.651
- baseline_factual: R_V=0.729
- baseline_math: R_V=0.741

## Overall Verdict

**PARTIAL CORRELATION - Investigate confounds**
