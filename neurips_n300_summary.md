# NeurIPS n=300 Experiment: Robust Behavior Transfer

**Date:** 2025-12-12 11:43:46.679121
**N:** 300 prompt pairs
**Method:** Full KV cache + Persistent V_PROJ at L27

## Results

### Behavior Scores

- Baseline: 0.76 ± 1.48
- Transfer: 2.62 ± 2.69
- Random control: 0.80 ± 1.58
- Wrong layer: 2.61 ± 2.62

### Transfer Effects

**Transfer:** Δ = 1.87 ± 2.95
- 95% CI: [1.53, 2.20]
- t(299) = 10.96, p = 9.89e-24
- Cohen's d = 0.63

**Random control:** Δ = 0.04 ± 1.95
- 95% CI: [-0.18, 0.26]
- t(299) = 0.36, p = 7.22e-01
- Cohen's d = 0.02

**Wrong layer:** Δ = 1.85 ± 2.86
- 95% CI: [1.52, 2.18]
- t(299) = 11.20, p = 1.54e-24
- Cohen's d = 0.65

### Comparisons

- Transfer vs Random: t = 8.95, p = 4.35e-18
- Transfer vs Wrong: t = 0.07, p = 9.44e-01
