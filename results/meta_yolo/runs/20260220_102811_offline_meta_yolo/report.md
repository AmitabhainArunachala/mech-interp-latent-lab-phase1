# Offline YOLO Meta-Experiment Report

- Generated: `2026-02-20T10:28:11.293052`
- Candidate finding selected: `cross_arch_universal_contraction`

## Project Index
- Total files scanned: 2793
- Pipeline Python files: 61
- Config JSON files: 263
- Results summary.json files: 246

## Experiment A: Cross-Architecture Meta
- Status: ok
- Latest models: 6 (negative deltas: 6)
- Sign test p-value: 0.015625

## Experiment B: Bridge Specificity Controls
- Status: ok

## Experiment C: Multi-Token Truncation Stress
- Status: ok
- Points analyzed: 14
- Truncation vs H3 rho: -0.6063348416289592
- Truncation vs H3 p-value: 0.021521097008774158
- H3 significance rate @ high truncation: 0.875
- H3 significance rate @ low truncation: 0.3333333333333333

## Candidate Ranking
1. cross_arch_universal_contraction (score=7.8061799739838875)
   evidence={'n_models': 6, 'sign_test_p': 0.015625, 'random_effect_mean_delta': nan}
2. truncation_driven_h3_instability (score=6.473470436497319)
   evidence={'n_points': 14, 'truncation_vs_h3_r': -0.6063348416289592, 'truncation_vs_h3_p': 0.021521097008774158}
3. gqa_headspace_specificity_bridge (score=0.0)
   evidence={'head_vs_random_p': nan, 'head_vs_baseline_donor_p': nan, 'head_vs_random_d': nan, 'head_vs_baseline_donor_d': nan, 'random_head_sign_flip_v2_to_v4': None}
