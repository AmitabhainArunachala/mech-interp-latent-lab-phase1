# Run report: rv_l27_activation_patching_bridge

- **run_dir**: `results/phase1_mechanism/runs/20260220_091409_rv_l27_activation_patching_bridge_seed_bridge_20260220_head_specific_s456_n80`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "config": "results/phase1_mechanism/runs/20260220_091409_rv_l27_activation_patching_bridge_seed_bridge_20260220_head_specific_s456_n80/config.json",
    "manifest": "results/phase1_mechanism/runs/20260220_091409_rv_l27_activation_patching_bridge_seed_bridge_20260220_head_specific_s456_n80/manifest.json",
    "per_sample_csv": "results/phase1_mechanism/runs/20260220_091409_rv_l27_activation_patching_bridge_seed_bridge_20260220_head_specific_s456_n80/per_sample.csv",
    "report": "results/phase1_mechanism/runs/20260220_091409_rv_l27_activation_patching_bridge_seed_bridge_20260220_head_specific_s456_n80/report.md",
    "summary": "results/phase1_mechanism/runs/20260220_091409_rv_l27_activation_patching_bridge_seed_bridge_20260220_head_specific_s456_n80/summary.json"
  },
  "baseline_groups": [
    "long_control",
    "baseline_creative",
    "baseline_math"
  ],
  "behavior_l4_count_delta_mean": 0.0,
  "behavior_strict_delta_mean": 0.013333333333333332,
  "behavior_strict_p_value": 0.6197661996270842,
  "behavior_word_count_delta_mean": 4.283333333333333,
  "device": "cuda",
  "do_sample": false,
  "donor_type": "recursive",
  "early_layer": 5,
  "experiment": "rv_l27_activation_patching_bridge",
  "head_space": "kv",
  "logit_diff_cohens_d": 0.33391461848586607,
  "logit_diff_delta_mean": 1.2020833333333334,
  "logit_diff_p_value": 0.012182613636844756,
  "max_length": 512,
  "max_new_tokens": 300,
  "model": "mistralai/Mistral-7B-v0.1",
  "n_pairs": 60,
  "n_skipped_short_baseline": 0,
  "n_truncated_baseline": 51,
  "n_truncated_patched": 54,
  "patch_heads_requested": [
    2
  ],
  "patch_kv_heads_effective": [
    2
  ],
  "patch_mode": "head_specific",
  "prompt_bank_version": "75e7c1b8dcebc24e",
  "recursive_groups": [
    "L5_refined",
    "L4_full",
    "L3_deeper"
  ],
  "rv_baseline_mean": 0.6870039002793732,
  "rv_cohens_d": -0.6510631978657235,
  "rv_delta_mean": -0.02541159436609395,
  "rv_p_value": 4.656245170367849e-06,
  "rv_patched_mean": 0.6615923059132793,
  "rv_recursive_mean": 0.5061580842872954,
  "schema_version": "metrics_summary_v1",
  "seed": 456,
  "target_layer": 27,
  "temperature": 0.0,
  "timestamp": "20260220_094913",
  "top_p": 0.95,
  "v_head_dim": 128,
  "v_num_heads": 8,
  "version": "v4_gqa_headspace",
  "window": 16
}
```
