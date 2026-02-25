# Offline YOLO Meta-Experiment Report

- Generated: `2026-02-20T10:28:38.470164`
- Candidate finding selected: `cross_arch_universal_contraction`

## Project Index
- Total files scanned: 2800
- Pipeline Python files: 61
- Config JSON files: 263
- Results summary.json files: 247

## Experiment A: Cross-Architecture Meta
- Status: ok
- Latest models: 6 (negative deltas: 6)
- Sign test p-value: 0.015625
- Random-effects mean delta: -0.1570030654050248
- Random-effects 95% CI: [-0.25449181851806024, -0.059514312291989335]
- Heterogeneity I2: 99.12729804527535

## Experiment B: Bridge Specificity Controls
- Status: ok
- v2_random_head_vs_v4_random_head: delta_mean=-0.045248501848064455, p=3.562916560633351e-11, d=-1.7404989908478083
- Random-head sign flip v2->v4: True (v2 mean=-0.0339909016422978, v4 mean=0.011257600205766652)

## Experiment C: Multi-Token Truncation Stress
- Status: ok
- Points analyzed: 14
- Truncation vs H3 rho: -0.6063348416289592
- Truncation vs H3 p-value: 0.021521097008774158
- H3 significance rate @ high truncation: 0.875
- H3 significance rate @ low truncation: 0.3333333333333333

## Candidate Ranking
1. cross_arch_universal_contraction (score=12.51627193613463)
   evidence={'n_models': 6, 'sign_test_p': 0.015625, 'random_effect_mean_delta': -0.1570030654050248}
2. truncation_driven_h3_instability (score=6.473470436497319)
   evidence={'n_points': 14, 'truncation_vs_h3_r': -0.6063348416289592, 'truncation_vs_h3_p': 0.021521097008774158}
3. gqa_headspace_specificity_bridge (score=2.0)
   evidence={'head_vs_random_p': nan, 'head_vs_baseline_donor_p': nan, 'head_vs_random_d': nan, 'head_vs_baseline_donor_d': nan, 'random_head_sign_flip_v2_to_v4': True}
