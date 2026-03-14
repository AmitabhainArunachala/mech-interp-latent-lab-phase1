# Run report: multi_token_bridge

- **run_dir**: `results/phase1_cross_architecture/runs/20260313_062646_multi_token_bridge_mistral_7b_bridge_true_longgen_n18`
- **prompt_bank_version**: `2ac959a313614329`

## Summary (machine-readable)

```json
{
  "analysis": {
    "temp_0.7": {
      "group_rv_means": {
        "L3_deeper": 0.5212680591991071,
        "L4_full": 0.4917384048746729,
        "L5_refined": 0.5023767185547414,
        "baseline_creative": 0.652340555625847,
        "baseline_math": 0.7423918926727505,
        "long_control": 0.6649000522755801
      },
      "group_word_means": {
        "L3_deeper": 941.3888888888889,
        "L4_full": 1271.8333333333333,
        "L5_refined": 1263.0555555555557,
        "baseline_creative": 1096.0555555555557,
        "baseline_math": 387.8888888888889,
        "long_control": 1149.9444444444443
      },
      "h1_all_spearman_p": 0.0003379805189747881,
      "h1_all_spearman_r": -0.3385937418167055,
      "h1_basis": "non_truncated",
      "h1_non_truncated_spearman_p": 0.821606939105165,
      "h1_non_truncated_spearman_r": 0.03632562397278182,
      "h1_significant": false,
      "h1_spearman_p": 0.821606939105165,
      "h1_spearman_r": 0.03632562397278182,
      "h2_baseline_rv_mean": 0.6865441668580593,
      "h2_cohens_d": 2.9749534036120178,
      "h2_p_value": 6.393328533218742e-29,
      "h2_recursive_rv_mean": 0.5051277275428405,
      "h2_significant": true,
      "h2_t_stat": -15.458311335617928,
      "h3_point_biserial_p": 0.016046896021229443,
      "h3_point_biserial_r": -0.2312365183657979,
      "h3_significant": false,
      "n_eos_reached": 41,
      "n_non_truncated": 41,
      "n_total": 108,
      "n_truncated": 67,
      "n_valid": 108,
      "pct_truncated": 62.03703703703704
    }
  },
  "artifacts": {
    "config": "results/phase1_cross_architecture/runs/20260313_062646_multi_token_bridge_mistral_7b_bridge_true_longgen_n18/config.json",
    "manifest": "results/phase1_cross_architecture/runs/20260313_062646_multi_token_bridge_mistral_7b_bridge_true_longgen_n18/manifest.json",
    "report": "results/phase1_cross_architecture/runs/20260313_062646_multi_token_bridge_mistral_7b_bridge_true_longgen_n18/report.md",
    "summary": "results/phase1_cross_architecture/runs/20260313_062646_multi_token_bridge_mistral_7b_bridge_true_longgen_n18/summary.json"
  },
  "baseline_groups": [
    "long_control",
    "baseline_creative",
    "baseline_math"
  ],
  "cohens_d": 2.9749534036120178,
  "early_layer": 5,
  "experiment": "multi_token_bridge",
  "late_layer": 27,
  "max_new_tokens": 2000,
  "model": "mistralai/Mistral-7B-v0.1",
  "n_pairs": 108,
  "n_prompts_per_group": 18,
  "n_total_prompts": 108,
  "p_value": 6.393328533218742e-29,
  "prompt_bank_version": "2ac959a313614329",
  "recursive_groups": [
    "L5_refined",
    "L4_full",
    "L3_deeper"
  ],
  "repetition_penalty": null,
  "rv_baseline_mean": 0.6865441668580593,
  "rv_cohens_d": 2.9749534036120178,
  "rv_delta_mean": 0.18141643931521878,
  "rv_p_value": 6.393328533218742e-29,
  "rv_recursive_mean": 0.5051277275428405,
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "temperatures": [
    0.7
  ],
  "timestamp": "20260313_070840",
  "top_p": 0.9,
  "version": "v2_gpt_audit_fixes",
  "window": 16
}
```
