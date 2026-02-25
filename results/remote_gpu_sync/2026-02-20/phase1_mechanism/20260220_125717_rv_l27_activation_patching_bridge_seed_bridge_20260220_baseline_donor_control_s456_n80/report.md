# Run report: rv_l27_activation_patching_bridge

- **run_dir**: `results/phase1_mechanism/runs/20260220_125717_rv_l27_activation_patching_bridge_seed_bridge_20260220_baseline_donor_control_s456_n80`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "config": "results/phase1_mechanism/runs/20260220_125717_rv_l27_activation_patching_bridge_seed_bridge_20260220_baseline_donor_control_s456_n80/config.json",
    "manifest": "results/phase1_mechanism/runs/20260220_125717_rv_l27_activation_patching_bridge_seed_bridge_20260220_baseline_donor_control_s456_n80/manifest.json",
    "pair_errors_csv": null,
    "per_sample_csv": "results/phase1_mechanism/runs/20260220_125717_rv_l27_activation_patching_bridge_seed_bridge_20260220_baseline_donor_control_s456_n80/per_sample.csv",
    "report": "results/phase1_mechanism/runs/20260220_125717_rv_l27_activation_patching_bridge_seed_bridge_20260220_baseline_donor_control_s456_n80/report.md",
    "summary": "results/phase1_mechanism/runs/20260220_125717_rv_l27_activation_patching_bridge_seed_bridge_20260220_baseline_donor_control_s456_n80/summary.json"
  },
  "baseline_groups": [
    "long_control",
    "baseline_creative",
    "baseline_math"
  ],
  "behavior_l4_count_delta_mean": 0.0,
  "behavior_strict_delta_mean": 0.021666666666666664,
  "behavior_strict_p_value": 0.3683003531986489,
  "behavior_word_count_delta_mean": 3.683333333333333,
  "device": "cuda",
  "do_sample": false,
  "donor_type": "baseline",
  "early_layer": 5,
  "experiment": "rv_l27_activation_patching_bridge",
  "generation_timeout_sec": 120,
  "head_space": "kv",
  "logit_diff_cohens_d": 0.33391461848586607,
  "logit_diff_delta_mean": 1.2020833333333334,
  "logit_diff_p_value": 0.012182613636844756,
  "max_length": 512,
  "max_new_tokens": 300,
  "model": "mistralai/Mistral-7B-v0.1",
  "n_pair_errors": 0,
  "n_pairs": 60,
  "n_skipped_short_baseline": 0,
  "n_truncated_baseline": 51,
  "n_truncated_patched": 53,
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
  "rv_cohens_d": 0.7077825276509184,
  "rv_delta_mean": 0.028964824336965584,
  "rv_p_value": 9.170227926437387e-07,
  "rv_patched_mean": 0.7159687246163389,
  "rv_recursive_mean": 0.5061580842872954,
  "schema_version": "metrics_summary_v1",
  "seed": 456,
  "target_layer": 27,
  "temperature": 0.0,
  "timestamp": "20260220_133157",
  "top_p": 0.95,
  "v_head_dim": 128,
  "v_num_heads": 8,
  "version": "v4_gqa_headspace",
  "window": 16
}
```
