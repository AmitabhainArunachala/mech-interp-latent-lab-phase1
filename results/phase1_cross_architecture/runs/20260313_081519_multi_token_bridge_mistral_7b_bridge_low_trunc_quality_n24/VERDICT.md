# Multi-Token Bridge Experiment Results (v2 - GPT Audit Fixes)

**Date**: 2026-03-13 08:33
**Model**: mistralai/Mistral-7B-v0.1
**N prompts**: 122 total
**Recursive groups**: ['L5_refined', 'L4_full', 'L3_deeper']
**Baseline groups**: ['long_control', 'baseline_creative', 'baseline_math']
**Seed**: 42

## Temperature 0.7

**Truncation**: 25/122 (20.5%) truncated, 97 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs recursive_content_score (non_truncated) | r=-0.415 | 2.38e-05 | Yes |
| H1 diagnostic: R_V vs word_count (non_truncated) | r=0.051 | 6.17e-01 | No |
| H1a: quality all valid | r=-0.449 | 2.08e-07 | Yes |
| H1b: quality non-truncated | r=-0.415 | 2.38e-05 | Yes |
| H1c: word-count all valid | r=0.008 | 9.32e-01 | No |
| H1d: word-count non-truncated | r=0.051 | 6.17e-01 | No |
| H2: Recursive vs Baseline R_V | d=2.91 | 1.19e-31 | Yes |
| H3: quality ordinal | r=-0.652 | 4.71e-13 | Yes |
| H4: BT+ART flag | r=-0.684 | 1.12e-14 | Yes |
| H5: L4 markers | r=-0.282 | 1.65e-03 | Yes |

**R_V means**: Recursive=0.506, Baseline=0.687

**Per-group R_V means**:
- L3_deeper: R_V=0.523
- L4_full: R_V=0.497
- L5_refined: R_V=0.497
- baseline_creative: R_V=0.651
- baseline_math: R_V=0.741
- long_control: R_V=0.669

## Overall Verdict

**STRONG CORRELATION - Proceed to sufficiency tests**
