# Final Correlation Summary
## Seed Bridge
- runs found: `9`
- complete seeds: `3` -> `[42, 123, 456]`
- standout gate seed passes: `[42, 123, 456]`
- `head_specific_vs_random_head_control`: mean_diff=-0.03838883203269895, p=3.467346019736592e-20, d=-0.7770406649337096, n_pairs=180
- `head_specific_vs_baseline_donor_control`: mean_diff=-0.05509582593134857, p=1.0965943100770243e-45, d=-1.440768889976374, n_pairs=180
- `random_head_control_vs_baseline_donor_control`: mean_diff=-0.0167069938986496, p=2.5364578961076668e-05, d=-0.3223167817945742, n_pairs=180

## Semantic (Seed Bridge)
- rows/runs/seeds: `540` / `9` / `[42, 123, 456]`
- semantic_recursive_rate by condition:
  - `baseline_donor_control`: rate=0.0, mean_score=0.171439762144453, n=180
  - `head_specific`: rate=0.0, mean_score=0.17212152329997885, n=180
  - `random_head_control`: rate=0.0, mean_score=0.16655172577334776, n=180
- Spearman rv_delta vs semantic_score: `{'n': 540, 'rho': 0.06902111011289148, 'p_value': 0.1091335035140809}`
- Spearman rv_patch vs semantic_score: `{'n': 540, 'rho': -0.1758909994179581, 'p_value': 3.9574368148300796e-05}`

## Semantic (C2)
- rows/sources: `765` / `11`
- Spearman rv_mean vs semantic_score: `{'n': 755, 'rho': -0.6519420205824219, 'p_value': 1.4298236437045998e-92}`
