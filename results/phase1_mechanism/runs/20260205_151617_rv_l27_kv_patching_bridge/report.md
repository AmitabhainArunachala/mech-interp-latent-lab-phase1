# Run report: rv_l27_kv_patching_bridge

- **run_dir**: `results/phase1_mechanism/runs/20260205_151617_rv_l27_kv_patching_bridge`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "config": "results/phase1_mechanism/runs/20260205_151617_rv_l27_kv_patching_bridge/config.json",
    "per_sample_csv": "results/phase1_mechanism/runs/20260205_151617_rv_l27_kv_patching_bridge/per_sample.csv",
    "report": "results/phase1_mechanism/runs/20260205_151617_rv_l27_kv_patching_bridge/report.md",
    "summary": "results/phase1_mechanism/runs/20260205_151617_rv_l27_kv_patching_bridge/summary.json"
  },
  "baseline_groups": [
    "long_control",
    "baseline_creative",
    "baseline_math"
  ],
  "behavior_l4_count_delta_mean": 0.15,
  "behavior_strict_delta_mean": -0.05,
  "behavior_strict_p_value": 0.4219239502739091,
  "behavior_word_count_delta_mean": 46.85,
  "device": "cuda",
  "do_sample": true,
  "early_layer": 5,
  "experiment": "rv_l27_kv_patching_bridge",
  "logit_diff_cohens_d": 0.2680426230935424,
  "logit_diff_delta_mean": 0.9384765625,
  "logit_diff_p_value": 0.24537360414703427,
  "max_length": 512,
  "max_new_tokens": 800,
  "model": "mistralai/Mistral-7B-v0.1",
  "n_pairs": 20,
  "n_skipped_short_baseline": 0,
  "n_truncated_baseline": 16,
  "n_truncated_patched": 18,
  "patch_window": 16,
  "prompt_bank_version": "75e7c1b8dcebc24e",
  "recursive_groups": [
    "L5_refined",
    "L4_full",
    "L3_deeper"
  ],
  "rv_baseline_mean": 0.6716631288548774,
  "rv_cohens_d": -1.8966002231046017,
  "rv_delta_mean": -0.1454003886602646,
  "rv_p_value": 6.955535180608606e-08,
  "rv_patched_mean": 0.5262627401946128,
  "rv_recursive_mean": 0.49381292446275005,
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "target_layer": 27,
  "temperature": 0.7,
  "timestamp": "20260205",
  "top_p": 0.95,
  "version": "v1",
  "window": 16
}
```
