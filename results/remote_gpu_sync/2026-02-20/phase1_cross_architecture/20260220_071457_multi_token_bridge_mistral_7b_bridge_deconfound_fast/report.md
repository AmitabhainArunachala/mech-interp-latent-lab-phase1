# Run report: multi_token_bridge

- **run_dir**: `results/phase1_cross_architecture/runs/20260220_071457_multi_token_bridge_mistral_7b_bridge_deconfound_fast`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "analysis": {
    "temp_0.0": {
      "group_rv_means": {
        "L3_deeper": 0.5241441649000208,
        "L4_full": 0.49712535795653406,
        "L5_refined": 0.4938837920048127,
        "baseline_creative": 0.6815467596148528,
        "baseline_math": 0.7421319811233041,
        "long_control": 0.6761004871169239
      },
      "group_word_means": {
        "L3_deeper": 487.8333333333333,
        "L4_full": 509.3333333333333,
        "L5_refined": 476.5,
        "baseline_creative": 451.3333333333333,
        "baseline_math": 140.5,
        "long_control": 439.0
      },
      "h1_significant": true,
      "h1_spearman_p": 1.7973489918200984e-05,
      "h1_spearman_r": -0.649803706520141,
      "h2_baseline_rv_mean": 0.6999264092850269,
      "h2_cohens_d": 3.5360731117568913,
      "h2_p_value": 2.5172823343618138e-12,
      "h2_recursive_rv_mean": 0.5050511049537891,
      "h2_significant": true,
      "h2_t_stat": -10.608219335270674,
      "h3_point_biserial_p": 0.3314183712788725,
      "h3_point_biserial_r": -0.1666218522590591,
      "h3_significant": false,
      "n_eos_reached": 4,
      "n_non_truncated": 4,
      "n_total": 36,
      "n_truncated": 32,
      "n_valid": 36,
      "pct_truncated": 88.88888888888889
    },
    "temp_0.7": {
      "group_rv_means": {
        "L3_deeper": 0.5241441649000208,
        "L4_full": 0.49712535795653406,
        "L5_refined": 0.4938837920048127,
        "baseline_creative": 0.6815467596148528,
        "baseline_math": 0.7421319811233041,
        "long_control": 0.6761004871169239
      },
      "group_word_means": {
        "L3_deeper": 401.6666666666667,
        "L4_full": 334.8333333333333,
        "L5_refined": 378.0,
        "baseline_creative": 461.0,
        "baseline_math": 118.66666666666667,
        "long_control": 411.8333333333333
      },
      "h1_significant": false,
      "h1_spearman_p": 0.21154501034209602,
      "h1_spearman_r": -0.4090909090909091,
      "h2_baseline_rv_mean": 0.6999264092850269,
      "h2_cohens_d": 3.5360731117568913,
      "h2_p_value": 2.5172823343618138e-12,
      "h2_recursive_rv_mean": 0.5050511049537891,
      "h2_significant": true,
      "h2_t_stat": -10.608219335270674,
      "h3_point_biserial_p": NaN,
      "h3_point_biserial_r": NaN,
      "h3_significant": false,
      "n_eos_reached": 11,
      "n_non_truncated": 11,
      "n_total": 36,
      "n_truncated": 25,
      "n_valid": 36,
      "pct_truncated": 69.44444444444444
    }
  },
  "artifacts": {
    "config": "results/phase1_cross_architecture/runs/20260220_071457_multi_token_bridge_mistral_7b_bridge_deconfound_fast/config.json",
    "report": "results/phase1_cross_architecture/runs/20260220_071457_multi_token_bridge_mistral_7b_bridge_deconfound_fast/report.md",
    "summary": "results/phase1_cross_architecture/runs/20260220_071457_multi_token_bridge_mistral_7b_bridge_deconfound_fast/summary.json"
  },
  "baseline_groups": [
    "long_control",
    "baseline_creative",
    "baseline_math"
  ],
  "cohens_d": 3.5360731117568913,
  "early_layer": 5,
  "experiment": "multi_token_bridge",
  "late_layer": 27,
  "max_new_tokens": 600,
  "model": "mistralai/Mistral-7B-v0.1",
  "n_pairs": 36,
  "n_prompts_per_group": 6,
  "n_total_prompts": 36,
  "p_value": 2.5172823343618138e-12,
  "prompt_bank_version": "75e7c1b8dcebc24e",
  "recursive_groups": [
    "L5_refined",
    "L4_full",
    "L3_deeper"
  ],
  "rv_baseline_mean": 0.6999264092850269,
  "rv_delta_mean": 0.1948753043312378,
  "rv_recursive_mean": 0.5050511049537891,
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "temperatures": [
    0.0,
    0.7
  ],
  "timestamp": "20260220_075333",
  "version": "v2_gpt_audit_fixes",
  "window": 16
}
```
