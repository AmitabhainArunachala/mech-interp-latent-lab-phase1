# Run report: rv_l27_activation_patching_bridge

- **run_dir**: `results/phase1_mechanism/runs/20260220_110921_rv_l27_activation_patching_bridge_seed_bridge_20260220_random_head_control_s123_n80`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "config": "results/phase1_mechanism/runs/20260220_110921_rv_l27_activation_patching_bridge_seed_bridge_20260220_random_head_control_s123_n80/config.json",
    "manifest": "results/phase1_mechanism/runs/20260220_110921_rv_l27_activation_patching_bridge_seed_bridge_20260220_random_head_control_s123_n80/manifest.json",
    "pair_errors_csv": null,
    "per_sample_csv": "results/phase1_mechanism/runs/20260220_110921_rv_l27_activation_patching_bridge_seed_bridge_20260220_random_head_control_s123_n80/per_sample.csv",
    "report": "results/phase1_mechanism/runs/20260220_110921_rv_l27_activation_patching_bridge_seed_bridge_20260220_random_head_control_s123_n80/report.md",
    "summary": "results/phase1_mechanism/runs/20260220_110921_rv_l27_activation_patching_bridge_seed_bridge_20260220_random_head_control_s123_n80/summary.json"
  },
  "baseline_groups": [
    "long_control",
    "baseline_creative",
    "baseline_math"
  ],
  "behavior_l4_count_delta_mean": 0.0,
  "behavior_strict_delta_mean": -0.019999999999999997,
  "behavior_strict_p_value": 0.4076699943617588,
  "behavior_word_count_delta_mean": 3.1333333333333333,
  "device": "cuda",
  "do_sample": false,
  "donor_type": "recursive",
  "early_layer": 5,
  "experiment": "rv_l27_activation_patching_bridge",
  "head_space": "kv",
  "logit_diff_cohens_d": 0.3396217674972204,
  "logit_diff_delta_mean": 1.2073567708333333,
  "logit_diff_p_value": 0.010853358018408053,
  "max_length": 512,
  "max_new_tokens": 300,
  "model": "mistralai/Mistral-7B-v0.1",
  "n_pair_errors": 0,
  "n_pairs": 60,
  "n_skipped_short_baseline": 0,
  "n_truncated_baseline": 51,
  "n_truncated_patched": 51,
  "patch_heads_requested": [
    2
  ],
  "patch_kv_heads_effective": [
    2
  ],
  "patch_mode": "random_head",
  "prompt_bank_version": "75e7c1b8dcebc24e",
  "recursive_groups": [
    "L5_refined",
    "L4_full",
    "L3_deeper"
  ],
  "rv_baseline_mean": 0.6870039002793732,
  "rv_cohens_d": 0.5467678294410098,
  "rv_delta_mean": 0.012510149452458881,
  "rv_p_value": 8.109784455294914e-05,
  "rv_patched_mean": 0.6995140497318322,
  "rv_recursive_mean": 0.5052559134512764,
  "schema_version": "metrics_summary_v1",
  "seed": 123,
  "target_layer": 27,
  "temperature": 0.0,
  "timestamp": "20260220_114743",
  "top_p": 0.95,
  "v_head_dim": 128,
  "v_num_heads": 8,
  "version": "v4_gqa_headspace",
  "window": 16
}
```
