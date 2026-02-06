# Multi-Token Bridge Experiment Results (v2 - GPT Audit Fixes)

**Date**: 2026-02-05 11:55
**Model**: mistralai/Mistral-7B-v0.1
**N prompts**: 85 total
**Recursive groups**: ['L1_hint', 'recursive_self_reference', 'introspective_concrete']
**Baseline groups**: ['baseline_personal', 'baseline_impossible', 'baseline_instructional']
**Seed**: 42

## Temperature 0.0

**Truncation**: 76/85 (89.4%) truncated, 9 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count | r=0.617 | 7.69e-02 | No |
| H2: Recursive vs Baseline R_V | d=0.10 | 6.89e-01 | No |
| H3: L4 markers | r=nan | nan | No |

**R_V means**: Recursive=0.653, Baseline=0.665

**Per-group R_V means**:
- L1_hint: R_V=0.596
- baseline_impossible: R_V=0.665
- baseline_instructional: R_V=0.717
- baseline_personal: R_V=0.652
- introspective_concrete: R_V=0.780
- recursive_self_reference: R_V=0.819

## Temperature 0.7

**Truncation**: 63/85 (74.1%) truncated, 22 hit EOS

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count | r=0.223 | 3.30e-01 | No |
| H2: Recursive vs Baseline R_V | d=0.10 | 6.89e-01 | No |
| H3: L4 markers | r=0.023 | 8.52e-01 | No |

**R_V means**: Recursive=0.653, Baseline=0.665

**Per-group R_V means**:
- L1_hint: R_V=0.596
- baseline_impossible: R_V=0.665
- baseline_instructional: R_V=0.717
- baseline_personal: R_V=0.652
- introspective_concrete: R_V=0.780
- recursive_self_reference: R_V=0.819

## Overall Verdict

**NO CORRELATION - R_V does not predict behavior**
