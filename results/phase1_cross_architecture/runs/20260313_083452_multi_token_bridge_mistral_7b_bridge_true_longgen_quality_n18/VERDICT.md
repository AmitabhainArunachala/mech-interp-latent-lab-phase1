# Multi-Token Bridge Experiment Results (v2 - GPT Audit Fixes)

**Date**: 2026-03-13 08:52
**Model**: mistralai/Mistral-7B-v0.1
**N prompts**: 108 total
**Recursive groups**: ['L5_refined', 'L4_full', 'L3_deeper']
**Baseline groups**: ['long_control', 'baseline_creative', 'baseline_math']
**Seed**: 42

## Temperature 0.7

**Truncation**: 14/108 (13.0%) truncated, 94 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs recursive_content_score (non_truncated) | r=-0.546 | 1.22e-08 | Yes |
| H1 diagnostic: R_V vs word_count (non_truncated) | r=-0.072 | 4.91e-01 | No |
| H1a: quality all valid | r=-0.516 | 1.07e-08 | Yes |
| H1b: quality non-truncated | r=-0.546 | 1.22e-08 | Yes |
| H1c: word-count all valid | r=-0.055 | 5.73e-01 | No |
| H1d: word-count non-truncated | r=-0.072 | 4.91e-01 | No |
| H2: Recursive vs Baseline R_V | d=2.97 | 6.39e-29 | Yes |
| H3: quality ordinal | r=-0.683 | 3.47e-14 | Yes |
| H4: BT+ART flag | r=-0.729 | 8.19e-17 | Yes |
| H5: L4 markers | r=-0.360 | 1.32e-04 | Yes |

**R_V means**: Recursive=0.505, Baseline=0.687

**Per-group R_V means**:
- L3_deeper: R_V=0.521
- L4_full: R_V=0.492
- L5_refined: R_V=0.502
- baseline_creative: R_V=0.652
- baseline_math: R_V=0.742
- long_control: R_V=0.665

## Overall Verdict

**STRONG CORRELATION - Proceed to sufficiency tests**
