# Run report: rv_l27_causal_validation

- **run_dir**: `results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "config": "results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/config.json",
    "pairs_csv": "results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/rv_l27_causal_validation_pairs.csv",
    "report": "results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/report.md",
    "summary": "results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/summary.json"
  },
  "delta_main": {
    "ci_95": [
      -0.14704411813535714,
      -0.060333123511823315
    ],
    "mean": -0.10368862082359023,
    "n": 45.0,
    "std": 0.14430991212237354
  },
  "delta_random": {
    "ci_95": [
      1.0156235638122693,
      1.1153936339032142
    ],
    "mean": 1.0655085988577417,
    "n": 45.0,
    "std": 0.16604365005587965
  },
  "delta_shuffled": {
    "ci_95": [
      -0.14704411813535845,
      -0.06033312351182486
    ],
    "mean": -0.10368862082359166,
    "n": 45.0,
    "std": 0.14430991212237318
  },
  "delta_wronglayer": {
    "ci_95": [
      -0.05888633617742804,
      0.006759768267262527
    ],
    "mean": -0.02606328395508276,
    "n": 45.0,
    "std": 0.10925239186471429
  },
  "device": "cuda",
  "experiment": "rv_l27_causal_validation",
  "logit_diff_cohens_d": null,
  "logit_diff_delta_mean": null,
  "logit_diff_p_value": null,
  "model": "Qwen/Qwen2.5-7B",
  "model_name": "Qwen/Qwen2.5-7B",
  "n_pairs": 45,
  "notes": {
    "measurement": "main/random/shuffled measured at target_layer; wronglayer measured at wrong_layer"
  },
  "params": {
    "early_layer": 4,
    "max_length": 512,
    "max_pairs": 45,
    "measure_target_after_wrong_patch": false,
    "pairing": {
      "baseline_groups": [
        "long_control",
        "baseline_creative",
        "baseline_math"
      ],
      "recursive_groups": [
        "L5_refined",
        "L4_full",
        "L3_deeper"
      ]
    },
    "target_layer": 24,
    "window": 16,
    "wrong_layer": 18
  },
  "prompt_bank_version": "75e7c1b8dcebc24e",
  "rv_baseline": {
    "ci_95": [
      1.222275256890837,
      1.2901264510977282
    ],
    "mean": 1.2562008539942826,
    "n": 45.0,
    "std": 0.11292224147475108
  },
  "rv_baseline_mean": 1.2562008539942826,
  "rv_cohens_d": -0.7185135054040029,
  "rv_delta_mean": -0.10368862082359023,
  "rv_p_value": 8.717424382925377e-06,
  "rv_recursive": {
    "ci_95": [
      1.1329313904216818,
      1.181958595394319
    ],
    "mean": 1.1574449929080004,
    "n": 45.0,
    "std": 0.08159417005795282
  },
  "rv_recursive_mean": 1.1574449929080004,
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "tests": {
    "main_effect_ttest_1samp_less_0": {
      "cohens_d": -0.7185135054040029,
      "n": 45,
      "p": 8.717424382925377e-06,
      "t": -4.819935122505039
    },
    "main_vs_random_paired_ttest": {
      "cohens_d": -8.073240209408437,
      "n": 45.0,
      "p": 6.512460880172992e-42,
      "t": -54.15694172076571
    },
    "main_vs_shuffled_paired_ttest": {
      "cohens_d": 0.2723150488251708,
      "n": 45.0,
      "p": 0.07452681834377285,
      "t": 1.8267448814077685
    },
    "main_vs_wronglayer_paired_ttest": {
      "cohens_d": -0.7074815378452249,
      "n": 45.0,
      "p": 2.2207672291716976e-05,
      "t": -4.745930434344039
    }
  },
  "timestamp": "20260202_120959",
  "transfer_percent_estimate": 104.99490327262518
}
```
