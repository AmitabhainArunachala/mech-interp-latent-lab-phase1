# Run report: rv_l27_causal_validation

- **run_dir**: `results/phase1_cross_architecture/runs/20260209_025948_rv_l27_causal_validation_gpt2_xl`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "config": "results/phase1_cross_architecture/runs/20260209_025948_rv_l27_causal_validation_gpt2_xl/config.json",
    "pairs_csv": "results/phase1_cross_architecture/runs/20260209_025948_rv_l27_causal_validation_gpt2_xl/rv_l27_causal_validation_pairs.csv",
    "report": "results/phase1_cross_architecture/runs/20260209_025948_rv_l27_causal_validation_gpt2_xl/report.md",
    "summary": "results/phase1_cross_architecture/runs/20260209_025948_rv_l27_causal_validation_gpt2_xl/summary.json"
  },
  "delta_main": {
    "ci_95": [
      -0.17373962660892855,
      -0.10134829091346746
    ],
    "mean": -0.13754395876119802,
    "n": 45.0,
    "std": 0.12047823160130056
  },
  "delta_random": {
    "ci_95": [
      0.7024133265550285,
      0.7805582052638598
    ],
    "mean": 0.7414857659094441,
    "n": 45.0,
    "std": 0.13005364115872242
  },
  "delta_shuffled": {
    "ci_95": [
      -0.17373962660892786,
      -0.10134829091346723
    ],
    "mean": -0.13754395876119754,
    "n": 45.0,
    "std": 0.12047823160129974
  },
  "delta_wronglayer": {
    "ci_95": [
      -0.03701756064982412,
      0.042510432653486356
    ],
    "mean": 0.0027464360018311144,
    "n": 45.0,
    "std": 0.13235550779571634
  },
  "device": "cuda",
  "experiment": "rv_l27_causal_validation",
  "logit_diff_cohens_d": null,
  "logit_diff_delta_mean": null,
  "logit_diff_p_value": null,
  "model": "openai-community/gpt2-xl",
  "model_name": "openai-community/gpt2-xl",
  "n_pairs": 45,
  "notes": {
    "measurement": "main/random/shuffled measured at target_layer; wronglayer measured at wrong_layer"
  },
  "params": {
    "early_layer": 6,
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
    "target_layer": 40,
    "window": 16,
    "wrong_layer": 32
  },
  "prompt_bank_version": "75e7c1b8dcebc24e",
  "rv_baseline": {
    "ci_95": [
      0.8201348206628464,
      0.8818327801525226
    ],
    "mean": 0.8509838004076845,
    "n": 45.0,
    "std": 0.10268163974754432
  },
  "rv_baseline_mean": 0.8509838004076845,
  "rv_cohens_d": -1.1416498809210047,
  "rv_delta_mean": -0.13754395876119802,
  "rv_p_value": 6.27139972063576e-10,
  "rv_recursive": {
    "ci_95": [
      0.7450590438254505,
      0.7891731586052584
    ],
    "mean": 0.7671161012153545,
    "n": 45.0,
    "std": 0.07341749515006232
  },
  "rv_recursive_mean": 0.7671161012153545,
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "tests": {
    "main_effect_ttest_1samp_less_0": {
      "cohens_d": -1.1416498809210047,
      "n": 45,
      "p": 6.27139972063576e-10,
      "t": -7.658420220731721
    },
    "main_vs_random_paired_ttest": {
      "cohens_d": -7.396713916832927,
      "n": 45.0,
      "p": 2.8823990663988594e-40,
      "t": -49.618665384471456
    },
    "main_vs_shuffled_paired_ttest": {
      "cohens_d": -0.11491372792119203,
      "n": 45.0,
      "p": 0.4449069747688956,
      "t": -0.770864721539103
    },
    "main_vs_wronglayer_paired_ttest": {
      "cohens_d": -2.758919651519025,
      "n": 45.0,
      "p": 2.192140889109013e-22,
      "t": -18.507395655769713
    }
  },
  "timestamp": "20260209_030002",
  "transfer_percent_estimate": 164.00111137635292
}
```
