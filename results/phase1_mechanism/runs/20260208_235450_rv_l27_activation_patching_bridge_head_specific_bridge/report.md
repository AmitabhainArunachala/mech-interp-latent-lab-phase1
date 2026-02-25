# Run report: rv_l27_activation_patching_bridge

- **run_dir**: `results/phase1_mechanism/runs/20260208_235450_rv_l27_activation_patching_bridge_head_specific_bridge`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "config": "results/phase1_mechanism/runs/20260208_235450_rv_l27_activation_patching_bridge_head_specific_bridge/config.json",
    "per_sample_csv": "results/phase1_mechanism/runs/20260208_235450_rv_l27_activation_patching_bridge_head_specific_bridge/per_sample.csv",
    "report": "results/phase1_mechanism/runs/20260208_235450_rv_l27_activation_patching_bridge_head_specific_bridge/report.md",
    "summary": "results/phase1_mechanism/runs/20260208_235450_rv_l27_activation_patching_bridge_head_specific_bridge/summary.json"
  },
  "baseline_groups": [
    "long_control",
    "baseline_creative",
    "baseline_math"
  ],
  "behavior_l4_count_delta_mean": -0.025,
  "behavior_strict_delta_mean": -0.0125,
  "behavior_strict_p_value": 0.6178560793762113,
  "behavior_word_count_delta_mean": -19.95,
  "device": "cuda",
  "do_sample": false,
  "early_layer": 5,
  "experiment": "rv_l27_activation_patching_bridge",
  "logit_diff_cohens_d": 0.3962257846348598,
  "logit_diff_delta_mean": 1.265234375,
  "logit_diff_p_value": 0.016493969171897466,
  "max_length": 512,
  "max_new_tokens": 800,
  "model": "mistralai/Mistral-7B-v0.1",
  "n_pairs": 40,
  "n_skipped_short_baseline": 0,
  "n_truncated_baseline": 27,
  "n_truncated_patched": 30,
  "patch_heads": [
    2,
    10,
    18,
    26
  ],
  "patch_mode": "head_specific",
  "prompt_bank_version": "75e7c1b8dcebc24e",
  "recursive_groups": [
    "L5_refined",
    "L4_full",
    "L3_deeper"
  ],
  "rv_baseline_mean": 0.6817947069296979,
  "rv_cohens_d": -0.7300873642673559,
  "rv_delta_mean": -0.024085171625152092,
  "rv_p_value": 4.1534309157980695e-05,
  "rv_patched_mean": 0.6577095353045458,
  "rv_recursive_mean": 0.5053887524206638,
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "target_layer": 27,
  "temperature": 0.0,
  "timestamp": "20260209_000856",
  "top_p": 0.95,
  "version": "v2_head_specific",
  "window": 16
}
```
