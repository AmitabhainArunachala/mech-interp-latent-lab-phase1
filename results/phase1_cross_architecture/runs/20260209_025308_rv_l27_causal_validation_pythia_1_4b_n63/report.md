# Run report: rv_l27_causal_validation

- **run_dir**: `results/phase1_cross_architecture/runs/20260209_025308_rv_l27_causal_validation_pythia_1_4b_n63`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "config": "results/phase1_cross_architecture/runs/20260209_025308_rv_l27_causal_validation_pythia_1_4b_n63/config.json",
    "pairs_csv": "results/phase1_cross_architecture/runs/20260209_025308_rv_l27_causal_validation_pythia_1_4b_n63/rv_l27_causal_validation_pairs.csv",
    "report": "results/phase1_cross_architecture/runs/20260209_025308_rv_l27_causal_validation_pythia_1_4b_n63/report.md",
    "summary": "results/phase1_cross_architecture/runs/20260209_025308_rv_l27_causal_validation_pythia_1_4b_n63/summary.json"
  },
  "delta_main": {
    "ci_95": [
      -0.008906961467269443,
      -0.001606785863370712
    ],
    "mean": -0.005256873665320078,
    "n": 63.0,
    "std": 0.014493289932175382
  },
  "delta_random": {
    "ci_95": [
      5.045379723080353,
      5.182544638908356
    ],
    "mean": 5.1139621809943545,
    "n": 63.0,
    "std": 0.27231822924313787
  },
  "delta_shuffled": {
    "ci_95": [
      -0.008906961467269424,
      -0.0016067858633706748
    ],
    "mean": -0.0052568736653200495,
    "n": 63.0,
    "std": 0.014493289932175416
  },
  "delta_wronglayer": {
    "ci_95": [
      -0.01129463612486787,
      -0.003945590794543333
    ],
    "mean": -0.007620113459705601,
    "n": 63.0,
    "std": 0.01459031268237019
  },
  "device": "cuda",
  "experiment": "rv_l27_causal_validation",
  "logit_diff_cohens_d": null,
  "logit_diff_delta_mean": null,
  "logit_diff_p_value": null,
  "model": "EleutherAI/pythia-1.4b",
  "model_name": "EleutherAI/pythia-1.4b",
  "n_pairs": 63,
  "notes": {
    "measurement": "main/random/shuffled measured at target_layer; wronglayer measured at wrong_layer"
  },
  "params": {
    "early_layer": 3,
    "max_length": 512,
    "max_pairs": 63,
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
    "target_layer": 20,
    "window": 16,
    "wrong_layer": 16
  },
  "prompt_bank_version": "75e7c1b8dcebc24e",
  "rv_baseline": {
    "ci_95": [
      0.37454375248926947,
      0.3839586136252881
    ],
    "mean": 0.3792511830572788,
    "n": 63.0,
    "std": 0.01869164791633419
  },
  "rv_baseline_mean": 0.3792511830572788,
  "rv_cohens_d": -0.36271086067558184,
  "rv_delta_mean": -0.005256873665320078,
  "rv_p_value": 0.002734688455674919,
  "rv_recursive": {
    "ci_95": [
      0.4155376343225649,
      0.4289024833693526
    ],
    "mean": 0.42222005884595876,
    "n": 63.0,
    "std": 0.026533694892408435
  },
  "rv_recursive_mean": 0.42222005884595876,
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "tests": {
    "main_effect_ttest_1samp_less_0": {
      "cohens_d": -0.36271086067558184,
      "n": 63,
      "p": 0.002734688455674919,
      "t": -2.8789282055093603
    },
    "main_vs_random_paired_ttest": {
      "cohens_d": -19.14640959317799,
      "n": 63.0,
      "p": 1.8349851631651737e-81,
      "t": -151.96991484999094
    },
    "main_vs_shuffled_paired_ttest": {
      "cohens_d": -0.10678897321252777,
      "n": 63.0,
      "p": 0.39991483423611995,
      "t": -0.8476111976528603
    },
    "main_vs_wronglayer_paired_ttest": {
      "cohens_d": 0.25085194122754517,
      "n": 63.0,
      "p": 0.05088394000589644,
      "t": 1.9910755571576253
    }
  },
  "timestamp": "20260209_025316",
  "transfer_percent_estimate": 0.0
}
```
