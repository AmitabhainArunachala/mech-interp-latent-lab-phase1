# Run report: multi_token_bridge

- **run_dir**: `results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "analysis": {
    "temp_0.0": {
      "group_rv_means": {
        "L3_deeper": 0.522943153523736,
        "L4_full": 0.4969613534385152,
        "L5_refined": 0.49690572817495215,
        "baseline_creative": 0.6509820310045182,
        "baseline_math": 0.7408063458952322,
        "long_control": 0.6692508132502077
      },
      "group_word_means": {
        "L3_deeper": 152.95,
        "L4_full": 160.95,
        "L5_refined": 158.2,
        "baseline_creative": 153.8,
        "baseline_math": 82.4,
        "long_control": 148.7
      },
      "h1_significant": false,
      "h1_spearman_p": 0.6368198117628943,
      "h1_spearman_r": -0.18333333333333335,
      "h2_baseline_rv_mean": 0.6870130633833196,
      "h2_cohens_d": 2.900164371363014,
      "h2_p_value": 4.376260212862936e-31,
      "h2_recursive_rv_mean": 0.5056034117124011,
      "h2_significant": true,
      "h2_t_stat": -15.884854466683121,
      "h3_point_biserial_p": 0.011625339725369253,
      "h3_point_biserial_r": -0.22966252163479717,
      "h3_significant": false,
      "n_eos_reached": 9,
      "n_non_truncated": 9,
      "n_total": 120,
      "n_truncated": 111,
      "n_valid": 120,
      "pct_truncated": 92.5
    },
    "temp_0.7": {
      "group_rv_means": {
        "L3_deeper": 0.522943153523736,
        "L4_full": 0.4969613534385152,
        "L5_refined": 0.49690572817495215,
        "baseline_creative": 0.6509820310045182,
        "baseline_math": 0.7408063458952322,
        "long_control": 0.6692508132502077
      },
      "group_word_means": {
        "L3_deeper": 136.9,
        "L4_full": 142.6,
        "L5_refined": 154.25,
        "baseline_creative": 149.0,
        "baseline_math": 85.0,
        "long_control": 146.1
      },
      "h1_significant": true,
      "h1_spearman_p": 0.0006200922160654913,
      "h1_spearman_r": -0.7608537747840488,
      "h2_baseline_rv_mean": 0.6870130633833196,
      "h2_cohens_d": 2.900164371363014,
      "h2_p_value": 4.376260212862936e-31,
      "h2_recursive_rv_mean": 0.5056034117124011,
      "h2_significant": true,
      "h2_t_stat": -15.884854466683121,
      "h3_point_biserial_p": 0.001521220936086485,
      "h3_point_biserial_r": -0.2863466703321508,
      "h3_significant": true,
      "n_eos_reached": 16,
      "n_non_truncated": 16,
      "n_total": 120,
      "n_truncated": 104,
      "n_valid": 120,
      "pct_truncated": 86.66666666666667
    }
  },
  "artifacts": {
    "config": "results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/config.json",
    "report": "results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/report.md",
    "summary": "results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/summary.json"
  },
  "baseline_groups": [
    "long_control",
    "baseline_creative",
    "baseline_math"
  ],
  "early_layer": 5,
  "experiment": "multi_token_bridge",
  "late_layer": 27,
  "max_new_tokens": 200,
  "model": "mistralai/Mistral-7B-v0.1",
  "n_prompts_per_group": 20,
  "n_total_prompts": 120,
  "prompt_bank_version": "75e7c1b8dcebc24e",
  "recursive_groups": [
    "L5_refined",
    "L4_full",
    "L3_deeper"
  ],
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "temperatures": [
    0.0,
    0.7
  ],
  "timestamp": "20260202_134439",
  "version": "v2_gpt_audit_fixes",
  "window": 16
}
```
