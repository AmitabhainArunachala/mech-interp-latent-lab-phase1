# Seed Bridge Analysis
- runs found: `9`
- complete seeds: `3`
- standout seed passes: `3` ([42, 123, 456])

## Per-seed contrasts
### Seed 42
- head_specific_vs_random_head_control [paired_t]: diff=-0.038566, p=1.78644e-08, d=-0.8410, overlap=60
- head_specific_vs_baseline_donor_control [paired_t]: diff=-0.055016, p=2.02862e-16, d=-1.4612, overlap=60
- random_head_control_vs_baseline_donor_control [paired_t]: diff=-0.016450, p=0.0283912, d=-0.2901, overlap=60

### Seed 123
- head_specific_vs_random_head_control [paired_t]: diff=-0.039068, p=1.74547e-07, d=-0.7645, overlap=60
- head_specific_vs_baseline_donor_control [paired_t]: diff=-0.055895, p=3.4703e-15, d=-1.3606, overlap=60
- random_head_control_vs_baseline_donor_control [paired_t]: diff=-0.016827, p=0.0112146, d=-0.3380, overlap=60

### Seed 456
- head_specific_vs_random_head_control [paired_t]: diff=-0.037533, p=5.75724e-07, d=-0.7238, overlap=60
- head_specific_vs_baseline_donor_control [paired_t]: diff=-0.054376, p=8.87302e-17, d=-1.4910, overlap=60
- random_head_control_vs_baseline_donor_control [paired_t]: diff=-0.016844, p=0.0108172, d=-0.3398, overlap=60

## Pooled random-effects
- head_specific_vs_random_head_control: mu=-0.038410, 95%CI=[-0.045624,-0.031196], I2=0.00%
- head_specific_vs_baseline_donor_control: mu=-0.055035, 95%CI=[-0.060624,-0.049447], I2=0.00%
- random_head_control_vs_baseline_donor_control: mu=-0.016729, 95%CI=[-0.024285,-0.009173], I2=0.00%

## Pooled paired t-tests
- head_specific_vs_random_head_control: mean_diff=-0.038389, p=3.46735e-20, d=-0.7770, n_pairs=180
- head_specific_vs_baseline_donor_control: mean_diff=-0.055096, p=1.09659e-45, d=-1.4408, n_pairs=180
- random_head_control_vs_baseline_donor_control: mean_diff=-0.016707, p=2.53646e-05, d=-0.3223, n_pairs=180
