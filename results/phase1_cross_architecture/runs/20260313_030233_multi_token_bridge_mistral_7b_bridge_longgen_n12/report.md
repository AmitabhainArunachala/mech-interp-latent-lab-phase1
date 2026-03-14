# Run report: multi_token_bridge

- **run_dir**: `results/phase1_cross_architecture/runs/20260313_030233_multi_token_bridge_mistral_7b_bridge_longgen_n12`
- **prompt_bank_version**: `2ac959a313614329`

## Summary (machine-readable)

```json
{
  "analysis": {
    "temp_0.0": {
      "group_rv_means": {
        "L3_deeper": 0.5271096799191538,
        "L4_full": 0.4858023851974098,
        "L5_refined": 0.49205885723228576,
        "baseline_creative": 0.6737482790556197,
        "baseline_math": 0.7517591618599556,
        "long_control": 0.6505261000921888
      },
      "group_word_means": {
        "L3_deeper": 906.5833333333334,
        "L4_full": 949.5833333333334,
        "L5_refined": 940.5,
        "baseline_creative": 925.0833333333334,
        "baseline_math": 205.0,
        "long_control": 805.1666666666666
      },
      "h1_all_spearman_p": 1.2774788633193112e-05,
      "h1_all_spearman_r": -0.48952814528031413,
      "h1_basis": "non_truncated",
      "h1_non_truncated_spearman_p": 0.4250381548921454,
      "h1_non_truncated_spearman_r": -0.28484848484848485,
      "h1_significant": false,
      "h1_spearman_p": 0.4250381548921454,
      "h1_spearman_r": -0.28484848484848485,
      "h2_baseline_rv_mean": 0.6920111803359215,
      "h2_cohens_d": 3.1673776166075878,
      "h2_p_value": 4.5910570390694446e-21,
      "h2_recursive_rv_mean": 0.5016569741162832,
      "h2_significant": true,
      "h2_t_stat": -13.43804514769026,
      "h3_point_biserial_p": 0.03589520849244466,
      "h3_point_biserial_r": -0.2477320918084467,
      "h3_significant": false,
      "n_eos_reached": 10,
      "n_non_truncated": 10,
      "n_total": 72,
      "n_truncated": 62,
      "n_valid": 72,
      "pct_truncated": 86.11111111111111
    },
    "temp_0.7": {
      "group_rv_means": {
        "L3_deeper": 0.5271096799191538,
        "L4_full": 0.4858023851974098,
        "L5_refined": 0.49205885723228576,
        "baseline_creative": 0.6737482790556197,
        "baseline_math": 0.7517591618599556,
        "long_control": 0.6505261000921888
      },
      "group_word_means": {
        "L3_deeper": 829.6666666666666,
        "L4_full": 852.0833333333334,
        "L5_refined": 737.8333333333334,
        "baseline_creative": 817.5,
        "baseline_math": 168.41666666666666,
        "long_control": 687.5
      },
      "h1_all_spearman_p": 0.0007981277560153194,
      "h1_all_spearman_r": -0.38649409121484235,
      "h1_basis": "non_truncated",
      "h1_non_truncated_spearman_p": 0.000561698520875261,
      "h1_non_truncated_spearman_r": -0.6883116883116883,
      "h1_significant": true,
      "h1_spearman_p": 0.000561698520875261,
      "h1_spearman_r": -0.6883116883116883,
      "h2_baseline_rv_mean": 0.6920111803359215,
      "h2_cohens_d": 3.1673776166075878,
      "h2_p_value": 4.5910570390694446e-21,
      "h2_recursive_rv_mean": 0.5016569741162832,
      "h2_significant": true,
      "h2_t_stat": -13.43804514769026,
      "h3_point_biserial_p": 0.007666766071917552,
      "h3_point_biserial_r": -0.31181950226580174,
      "h3_significant": true,
      "n_eos_reached": 21,
      "n_non_truncated": 21,
      "n_total": 72,
      "n_truncated": 51,
      "n_valid": 72,
      "pct_truncated": 70.83333333333333
    }
  },
  "artifacts": {
    "config": "results/phase1_cross_architecture/runs/20260313_030233_multi_token_bridge_mistral_7b_bridge_longgen_n12/config.json",
    "manifest": "results/phase1_cross_architecture/runs/20260313_030233_multi_token_bridge_mistral_7b_bridge_longgen_n12/manifest.json",
    "report": "results/phase1_cross_architecture/runs/20260313_030233_multi_token_bridge_mistral_7b_bridge_longgen_n12/report.md",
    "summary": "results/phase1_cross_architecture/runs/20260313_030233_multi_token_bridge_mistral_7b_bridge_longgen_n12/summary.json"
  },
  "baseline_groups": [
    "long_control",
    "baseline_creative",
    "baseline_math"
  ],
  "cohens_d": 3.1673776166075878,
  "early_layer": 5,
  "experiment": "multi_token_bridge",
  "late_layer": 27,
  "max_new_tokens": 1200,
  "model": "mistralai/Mistral-7B-v0.1",
  "n_pairs": 72,
  "n_prompts_per_group": 12,
  "n_total_prompts": 72,
  "p_value": 4.5910570390694446e-21,
  "prompt_bank_version": "2ac959a313614329",
  "recursive_groups": [
    "L5_refined",
    "L4_full",
    "L3_deeper"
  ],
  "rv_baseline_mean": 0.6920111803359215,
  "rv_cohens_d": 3.1673776166075878,
  "rv_delta_mean": 0.19035420621963828,
  "rv_p_value": 4.5910570390694446e-21,
  "rv_recursive_mean": 0.5016569741162832,
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "temperatures": [
    0.0,
    0.7
  ],
  "timestamp": "20260313_034031",
  "version": "v2_gpt_audit_fixes",
  "window": 16
}
```
