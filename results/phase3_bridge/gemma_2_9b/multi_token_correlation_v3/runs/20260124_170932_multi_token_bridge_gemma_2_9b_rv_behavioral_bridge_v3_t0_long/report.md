# Run report: multi_token_bridge

- **run_dir**: `results/phase3_bridge/gemma_2_9b/multi_token_correlation_v3/runs/20260124_170932_multi_token_bridge_gemma_2_9b_rv_behavioral_bridge_v3_t0_long`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "analysis": {
    "temp_0.0": {
      "group_rv_means": {
        "L3_deeper": 0.6073994676704797,
        "L4_full": 0.5921876392309031,
        "baseline_creative": 0.7705321513311778,
        "baseline_factual": 0.7945093587681862,
        "baseline_math": 0.7656868224399005,
        "champions": 0.6216247966023158
      },
      "group_word_means": {
        "L3_deeper": 350.09090909090907,
        "L4_full": 327.65,
        "baseline_creative": 314.65,
        "baseline_factual": 299.5,
        "baseline_math": 55.55,
        "champions": 325.0
      },
      "h1_significant": false,
      "h1_spearman_p": 0.497760923956493,
      "h1_spearman_r": -0.17089703174621093,
      "h2_baseline_rv_mean": 0.7769094441797552,
      "h2_cohens_d": 3.369167474507709,
      "h2_p_value": 1.1020946646967023e-35,
      "h2_recursive_rv_mean": 0.6058054916211115,
      "h2_significant": true,
      "h2_t_stat": -18.21556814943375,
      "h3_point_biserial_p": 0.008841967078217182,
      "h3_point_biserial_r": -0.24105181141878224,
      "h3_significant": true,
      "n_eos_reached": 18,
      "n_non_truncated": 18,
      "n_total": 117,
      "n_truncated": 99,
      "n_valid": 117,
      "pct_truncated": 84.61538461538461
    }
  },
  "artifacts": {
    "config": "results/phase3_bridge/gemma_2_9b/multi_token_correlation_v3/runs/20260124_170932_multi_token_bridge_gemma_2_9b_rv_behavioral_bridge_v3_t0_long/config.json",
    "report": "results/phase3_bridge/gemma_2_9b/multi_token_correlation_v3/runs/20260124_170932_multi_token_bridge_gemma_2_9b_rv_behavioral_bridge_v3_t0_long/report.md",
    "summary": "results/phase3_bridge/gemma_2_9b/multi_token_correlation_v3/runs/20260124_170932_multi_token_bridge_gemma_2_9b_rv_behavioral_bridge_v3_t0_long/summary.json"
  },
  "baseline_groups": [
    "baseline_factual",
    "baseline_math",
    "baseline_creative"
  ],
  "early_layer": 5,
  "experiment": "multi_token_bridge",
  "late_layer": 38,
  "max_new_tokens": 400,
  "model": "google/gemma-2-9b",
  "n_prompts_per_group": 40,
  "n_total_prompts": 117,
  "prompt_bank_version": "75e7c1b8dcebc24e",
  "recursive_groups": [
    "champions",
    "L4_full",
    "L3_deeper"
  ],
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "temperatures": [
    0.0
  ],
  "timestamp": "20260124_172729",
  "version": "v2_gpt_audit_fixes",
  "window": 16
}
```
