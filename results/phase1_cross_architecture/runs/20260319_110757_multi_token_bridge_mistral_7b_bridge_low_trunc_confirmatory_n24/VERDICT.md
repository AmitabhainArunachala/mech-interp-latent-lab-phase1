# Multi-Token Bridge Experiment Results (v2 - GPT Audit Fixes)

**Date**: 2026-03-19 13:13
**Model**: mistralai/Mistral-7B-v0.1
**N prompts**: 122 total
**Recursive groups**: ['L5_refined', 'L4_full', 'L3_deeper']
**Baseline groups**: ['long_control', 'baseline_creative', 'baseline_math']
**Seed**: 42

## Temperature 0.7

**Truncation**: 94/122 (77.0%) truncated, 28 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs recursive_content_score (non_truncated) | r=-0.015 | 9.38e-01 | No |
| H1 diagnostic: R_V vs word_count (non_truncated) | r=-0.292 | 1.32e-01 | No |
| H1a: quality all valid | r=0.041 | 6.52e-01 | No |
| H1b: quality non-truncated | r=-0.015 | 9.38e-01 | No |
| H1c: word-count all valid | r=-0.349 | 8.08e-05 | Yes |
| H1d: word-count non-truncated | r=-0.292 | 1.32e-01 | No |
| H2: Recursive vs Baseline R_V | d=2.91 | 1.17e-31 | Yes |
| H3: quality ordinal | r=-0.118 | 5.50e-01 | No |
| H4: BT+ART flag | r=-0.636 | 2.72e-04 | Yes |
| H5: L4 markers | r=-0.254 | 4.68e-03 | Yes |

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
