# Multi-Token Bridge Experiment Results (v2 - GPT Audit Fixes)

**Date**: 2026-03-19 15:57
**Model**: mistralai/Mistral-7B-v0.1
**N prompts**: 108 total
**Recursive groups**: ['L5_refined', 'L4_full', 'L3_deeper']
**Baseline groups**: ['long_control', 'baseline_creative', 'baseline_math']
**Seed**: 42

## Temperature 0.7

**Truncation**: 75/108 (69.4%) truncated, 33 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs recursive_content_score (non_truncated) | r=-0.401 | 2.08e-02 | No |
| H1 diagnostic: R_V vs word_count (non_truncated) | r=-0.043 | 8.10e-01 | No |
| H1a: quality all valid | r=-0.098 | 3.12e-01 | No |
| H1b: quality non-truncated | r=-0.401 | 2.08e-02 | No |
| H1c: word-count all valid | r=-0.464 | 4.12e-07 | Yes |
| H1d: word-count non-truncated | r=-0.043 | 8.10e-01 | No |
| H2: Recursive vs Baseline R_V | d=2.98 | 6.36e-29 | Yes |
| H3: quality ordinal | r=-0.345 | 4.91e-02 | No |
| H4: BT+ART flag | r=-0.677 | 1.50e-05 | Yes |
| H5: L4 markers | r=-0.374 | 6.56e-05 | Yes |

**R_V means**: Recursive=0.505, Baseline=0.687

**Per-group R_V means**:
- L3_deeper: R_V=0.521
- L4_full: R_V=0.492
- L5_refined: R_V=0.502
- baseline_creative: R_V=0.652
- baseline_math: R_V=0.742
- long_control: R_V=0.665

## Overall Verdict

**PARTIAL CORRELATION - Investigate confounds**
