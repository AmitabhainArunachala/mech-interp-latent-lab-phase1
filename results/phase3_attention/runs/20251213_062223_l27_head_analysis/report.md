# L27 Critical Head Analysis Report

**Model:** mistralai/Mistral-7B-v0.1
**Target Layer:** 27
**Critical Heads:** H11, H1, H22 (from ablation study)
**Control Head:** H5

## R_V Values
- Champion: **0.4549**
- Baseline: **0.6102**

## Entropy Comparison
| Head | Champion | Baseline | Delta |
|------|----------|----------|-------|
| H11 | 0.738 | 0.820 | -0.082 |
| H1 | 0.510 | 0.455 | +0.055 |
| H22 | 0.802 | 0.648 | +0.154 |
| H5 | 0.458 | 0.252 | +0.206 |

## Self-Attention Ratio
| Head | Champion | Baseline | Delta |
|------|----------|----------|-------|
| H11 | 0.0587 | 0.0598 | -0.0011 |
| H1 | 0.0382 | 0.0505 | -0.0123 |
| H22 | 0.0498 | 0.0402 | +0.0097 |
| H5 | 0.0293 | 0.0259 | +0.0034 |

## Interpretation

**Entropy:** Lower entropy = more focused attention. If critical heads show lower entropy on champion prompts, they may be focusing on specific self-referential tokens.

**Self-attention:** Higher self-attention ratio means the head attends more to the current position. Changes here indicate different information routing.

## Artifacts
- CSV: `head_comparison.csv`
- Heatmaps: `attn_champion_H*.png`, `attn_baseline_H*.png`