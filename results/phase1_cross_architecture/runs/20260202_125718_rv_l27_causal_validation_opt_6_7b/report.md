# Run report: rv_l27_causal_validation

- **run_dir**: `results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "config": "results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/config.json",
    "pairs_csv": "results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/rv_l27_causal_validation_pairs.csv",
    "report": "results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/report.md",
    "summary": "results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/summary.json"
  },
  "delta_main": {
    "ci_95": [
      -0.4193002874005628,
      -0.3013668594501928
    ],
    "mean": -0.3603335734253778,
    "n": 45.0,
    "std": 0.19627225702689757
  },
  "delta_random": {
    "ci_95": [
      2.575255991560573,
      2.757228761892572
    ],
    "mean": 2.6662423767265726,
    "n": 45.0,
    "std": 0.30285057401646337
  },
  "delta_shuffled": {
    "ci_95": [
      -0.4193002874005613,
      -0.30136685945019154
    ],
    "mean": -0.3603335734253764,
    "n": 45.0,
    "std": 0.19627225702689727
  },
  "delta_wronglayer": {
    "ci_95": [
      0.18576232344335347,
      0.332767531439125
    ],
    "mean": 0.25926492744123925,
    "n": 45.0,
    "std": 0.2446553489497555
  },
  "device": "cuda",
  "experiment": "rv_l27_causal_validation",
  "logit_diff_cohens_d": null,
  "logit_diff_delta_mean": null,
  "logit_diff_p_value": null,
  "model": "facebook/opt-6.7b",
  "model_name": "facebook/opt-6.7b",
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
    "target_layer": 27,
    "window": 16,
    "wrong_layer": 21
  },
  "prompt_bank_version": "75e7c1b8dcebc24e",
  "rv_baseline": {
    "ci_95": [
      1.133196768652644,
      1.2674323537755985
    ],
    "mean": 1.2003145612141213,
    "n": 45.0,
    "std": 0.22340333630000112
  },
  "rv_baseline_mean": 1.2003145612141213,
  "rv_cohens_d": -1.8358864308367173,
  "rv_delta_mean": -0.3603335734253778,
  "rv_p_value": 3.7297415488741014e-16,
  "rv_recursive": {
    "ci_95": [
      0.906390020216925,
      0.9736221461910786
    ],
    "mean": 0.9400060832040018,
    "n": 45.0,
    "std": 0.11189194903430617
  },
  "rv_recursive_mean": 0.9400060832040018,
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "tests": {
    "main_effect_ttest_1samp_less_0": {
      "cohens_d": -1.8358864308367173,
      "n": 45,
      "p": 3.7297415488741014e-16,
      "t": -12.3155005749611
    },
    "main_vs_random_paired_ttest": {
      "cohens_d": -9.204370108432453,
      "n": 45.0,
      "p": 2.1879362322821781e-44,
      "t": -61.74479175756623
    },
    "main_vs_shuffled_paired_ttest": {
      "cohens_d": -0.22108248457986313,
      "n": 45.0,
      "p": 0.14518475954611396,
      "t": -1.4830663924653689
    },
    "main_vs_wronglayer_paired_ttest": {
      "cohens_d": -4.100336749428286,
      "n": 45.0,
      "p": 2.347453895914512e-29,
      "t": -27.505895107086506
    }
  },
  "timestamp": "20260202_125756",
  "transfer_percent_estimate": 138.4256003415186
}
```
