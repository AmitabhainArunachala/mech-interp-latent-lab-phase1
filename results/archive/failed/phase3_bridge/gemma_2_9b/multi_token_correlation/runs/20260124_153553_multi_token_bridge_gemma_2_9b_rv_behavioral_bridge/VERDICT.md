# Multi-Token Bridge Experiment Results
**Date**: 2026-01-24 15:46
**Model**: google/gemma-2-9b
**N prompts**: 60 (20 per group)

## Temperature 0.0

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count | r=-0.275 | 3.32e-02 | No |
| H2: L4 vs L3 R_V | d=0.40 | 2.12e-01 | No |
| H3: L4 markers | r=-0.226 | 8.20e-02 | No |

**R_V means**: L4=0.592, L3=0.608, baseline=0.795

**Word count means**: L4=165.8, L3=174.4, baseline=150.2

## Temperature 0.7

| Hypothesis | Statistic | p-value | Significant |
|------------|-----------|---------|-------------|
| H1: R_V vs word_count | r=-0.323 | 1.17e-02 | No |
| H2: L4 vs L3 R_V | d=0.40 | 2.12e-01 | No |
| H3: L4 markers | r=-0.274 | 3.44e-02 | No |

**R_V means**: L4=0.592, L3=0.608, baseline=0.795

**Word count means**: L4=162.4, L3=162.9, baseline=133.8

## Overall Verdict

**NO CORRELATION - R_V does not predict behavior**
