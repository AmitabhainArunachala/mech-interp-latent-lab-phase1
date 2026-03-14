# Run report: multi_token_bridge

- **run_dir**: `results/phase1_cross_architecture/runs/20260313_081519_multi_token_bridge_mistral_7b_bridge_low_trunc_quality_n24`
- **prompt_bank_version**: `2ac959a313614329`

## Summary (machine-readable)

```json
{
  "analysis": {
    "temp_0.7": {
      "group_quality_means": {
        "L3_deeper": 0.5476352188700594,
        "L4_full": 0.4787914074982213,
        "L5_refined": 0.5223319375222226,
        "baseline_creative": 0.37787828663093637,
        "baseline_math": 0.3141112636355661,
        "long_control": 0.4499885149499544
      },
      "group_rv_means": {
        "L3_deeper": 0.5234065831357122,
        "L4_full": 0.49693329904301553,
        "L5_refined": 0.4969412965793027,
        "baseline_creative": 0.6509918890100048,
        "baseline_math": 0.7407553005370116,
        "long_control": 0.6692463207799555
      },
      "group_word_means": {
        "L3_deeper": 484.1818181818182,
        "L4_full": 321.1,
        "L5_refined": 315.9,
        "baseline_creative": 366.05,
        "baseline_math": 216.45,
        "long_control": 611.9
      },
      "h1_all_spearman_p": 0.9321461845833385,
      "h1_all_spearman_r": 0.007788723040033782,
      "h1_basis": "non_truncated",
      "h1_non_truncated_spearman_p": 0.6173260043541914,
      "h1_non_truncated_spearman_r": 0.0513634794751121,
      "h1_quality_all_spearman_p": 2.0781416202102324e-07,
      "h1_quality_all_spearman_r": -0.44938388281051217,
      "h1_quality_basis": "non_truncated",
      "h1_quality_non_truncated_spearman_p": 2.377774019879069e-05,
      "h1_quality_non_truncated_spearman_r": -0.4149616031979802,
      "h1_quality_significant": true,
      "h1_quality_spearman_p": 2.377774019879069e-05,
      "h1_quality_spearman_r": -0.4149616031979802,
      "h1_significant": false,
      "h1_spearman_p": 0.6173260043541914,
      "h1_spearman_r": 0.0513634794751121,
      "h2_baseline_rv_mean": 0.6869978367756573,
      "h2_cohens_d": 2.906372426816336,
      "h2_p_value": 1.1865186021699605e-31,
      "h2_recursive_rv_mean": 0.506329624861807,
      "h2_significant": true,
      "h2_t_stat": -16.0488093992018,
      "h3_class_basis": "non_truncated",
      "h3_class_significant": true,
      "h3_class_spearman_p": 4.70758031421188e-13,
      "h3_class_spearman_r": -0.6519898751625309,
      "h3_point_biserial_p": 0.0016454307900425128,
      "h3_point_biserial_r": -0.28207574365099447,
      "h3_significant": true,
      "h4_bt_art_basis": "non_truncated",
      "h4_bt_art_pointbiserial_p": 1.1248920417867518e-14,
      "h4_bt_art_pointbiserial_r": -0.6841341870996879,
      "h4_bt_art_significant": true,
      "n_eos_reached": 97,
      "n_non_truncated": 97,
      "n_total": 122,
      "n_truncated": 25,
      "n_valid": 122,
      "pct_truncated": 20.491803278688526
    }
  },
  "artifacts": {
    "config": "results/phase1_cross_architecture/runs/20260313_081519_multi_token_bridge_mistral_7b_bridge_low_trunc_quality_n24/config.json",
    "manifest": "results/phase1_cross_architecture/runs/20260313_081519_multi_token_bridge_mistral_7b_bridge_low_trunc_quality_n24/manifest.json",
    "report": "results/phase1_cross_architecture/runs/20260313_081519_multi_token_bridge_mistral_7b_bridge_low_trunc_quality_n24/report.md",
    "summary": "results/phase1_cross_architecture/runs/20260313_081519_multi_token_bridge_mistral_7b_bridge_low_trunc_quality_n24/summary.json"
  },
  "baseline_groups": [
    "long_control",
    "baseline_creative",
    "baseline_math"
  ],
  "bt_art_basis": "non_truncated",
  "bt_art_pointbiserial_p": 1.1248920417867518e-14,
  "bt_art_pointbiserial_r": -0.6841341870996879,
  "cohens_d": 2.906372426816336,
  "early_layer": 5,
  "experiment": "multi_token_bridge",
  "late_layer": 27,
  "max_new_tokens": 1200,
  "model": "mistralai/Mistral-7B-v0.1",
  "n_pairs": 122,
  "n_prompts_per_group": 24,
  "n_total_prompts": 122,
  "p_value": 1.1865186021699605e-31,
  "prompt_bank_version": "2ac959a313614329",
  "quality_basis": "non_truncated",
  "quality_spearman_p": 2.377774019879069e-05,
  "quality_spearman_r": -0.4149616031979802,
  "recursive_groups": [
    "L5_refined",
    "L4_full",
    "L3_deeper"
  ],
  "repetition_penalty": 1.1,
  "rv_baseline_mean": 0.6869978367756573,
  "rv_cohens_d": 2.906372426816336,
  "rv_delta_mean": 0.1806682119138503,
  "rv_p_value": 1.1865186021699605e-31,
  "rv_recursive_mean": 0.506329624861807,
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "temperatures": [
    0.7
  ],
  "timestamp": "20260313_083352",
  "top_p": 0.9,
  "version": "v2_gpt_audit_fixes",
  "window": 16
}
```
