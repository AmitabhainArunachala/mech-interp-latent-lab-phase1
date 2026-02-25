# GPU Session Summary (2026-02-20)

## Bridge Matrix (Mistral-7B, fast)
Source: `results/remote_gpu_sync/2026-02-20/phase1_mechanism/contrast_stats.md`

- head_specific mean `rv_delta=-0.027630`
- random_head mean `rv_delta=0.023466`
- baseline_donor mean `rv_delta=0.029746`

Pairwise:
- head_specific vs random_head: `mean diff=-0.051096`, `p=4.30594e-04`, `d=-2.3657`
- head_specific vs baseline_donor: `mean diff=-0.057376`, `p=1.67042e-02`, `d=-1.4294`
- random_head vs baseline_donor: `mean diff=-0.006280`, `p=7.49631e-01`, `d=-0.1647`

## Multi-token Bridge Re-run (Mistral-7B)
Run: `results/remote_gpu_sync/2026-02-20/phase1_cross_architecture/20260220_071457_multi_token_bridge_mistral_7b_bridge_deconfound_fast/summary.json`

Temperature 0.0:
- truncation: `32/36 (88.9%)`
- H1 (`R_V` vs word_count): `r=-0.6498`, `p=1.797e-05`
- H2 (recursive vs baseline `R_V`): `d=3.5361`, `p=2.517e-12`

Temperature 0.7:
- truncation: `25/36 (69.4%)`
- H1 (`R_V` vs word_count): `r=-0.4091`, `p=0.2115`
- H2 (recursive vs baseline `R_V`): `d=3.5361`, `p=2.517e-12`

## Interpretation
- Geometry-side specificity claim is supported by bridge controls.
- Behavior-side claims remain truncation-sensitive and require low-truncation confirmatory runs.
