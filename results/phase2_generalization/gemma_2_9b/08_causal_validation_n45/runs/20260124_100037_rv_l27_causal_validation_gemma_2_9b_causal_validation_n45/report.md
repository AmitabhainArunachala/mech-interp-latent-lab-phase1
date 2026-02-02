# Run report: rv_l27_causal_validation

- **run_dir**: `results/phase2_generalization/gemma_2_9b/08_causal_validation_n45/runs/20260124_100037_rv_l27_causal_validation_gemma_2_9b_causal_validation_n45`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "config": "results/phase2_generalization/gemma_2_9b/08_causal_validation_n45/runs/20260124_100037_rv_l27_causal_validation_gemma_2_9b_causal_validation_n45/config.json",
    "pairs_csv": "results/phase2_generalization/gemma_2_9b/08_causal_validation_n45/runs/20260124_100037_rv_l27_causal_validation_gemma_2_9b_causal_validation_n45/rv_l27_causal_validation_pairs.csv",
    "report": "results/phase2_generalization/gemma_2_9b/08_causal_validation_n45/runs/20260124_100037_rv_l27_causal_validation_gemma_2_9b_causal_validation_n45/report.md",
    "summary": "results/phase2_generalization/gemma_2_9b/08_causal_validation_n45/runs/20260124_100037_rv_l27_causal_validation_gemma_2_9b_causal_validation_n45/summary.json"
  },
  "delta_main": {
    "ci_95": [
      -0.19412336473682532,
      -0.14130852112907602
    ],
    "mean": -0.16771594293295067,
    "n": 45.0,
    "std": 0.08789779742328813
  },
  "delta_random": {
    "ci_95": [
      1.1993821030504568,
      1.3147330105702368
    ],
    "mean": 1.2570575568103468,
    "n": 45.0,
    "std": 0.1919742634678258
  },
  "delta_shuffled": {
    "ci_95": [
      -0.19412336473682504,
      -0.14130852112907596
    ],
    "mean": -0.1677159429329505,
    "n": 45.0,
    "std": 0.08789779742328781
  },
  "delta_wronglayer": {
    "ci_95": [
      -0.05541884238708326,
      0.0031702002979906996
    ],
    "mean": -0.02612432104454628,
    "n": 45.0,
    "std": 0.09750758410655214
  },
  "device": "cuda",
  "experiment": "rv_l27_causal_validation",
  "logit_diff_cohens_d": null,
  "logit_diff_delta_mean": null,
  "logit_diff_p_value": null,
  "model": "google/gemma-2-9b",
  "model_name": "google/gemma-2-9b",
  "n_pairs": 45,
  "notes": {
    "measurement": "main/random/shuffled measured at target_layer; wronglayer measured at wrong_layer"
  },
  "params": {
    "early_layer": 5,
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
    "target_layer": 38,
    "window": 16,
    "wrong_layer": 20
  },
  "prompt_bank_version": "75e7c1b8dcebc24e",
  "rv_baseline": {
    "ci_95": [
      0.7552270989595299,
      0.7880383086091668
    ],
    "mean": 0.7716327037843483,
    "n": 45.0,
    "std": 0.054606486775124335
  },
  "rv_baseline_mean": 0.7716327037843483,
  "rv_cohens_d": -1.9080790173305877,
  "rv_delta_mean": -0.16771594293295067,
  "rv_p_value": 9.818481781842745e-17,
  "rv_recursive": {
    "ci_95": [
      0.5785046573245077,
      0.6087771661671108
    ],
    "mean": 0.5936409117458092,
    "n": 45.0,
    "std": 0.05038142060031411
  },
  "rv_recursive_mean": 0.5936409117458092,
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "tests": {
    "main_effect_ttest_1samp_less_0": {
      "cohens_d": -1.9080790173305877,
      "n": 45,
      "p": 9.818481781842745e-17,
      "t": -12.799783167576578
    },
    "main_vs_random_paired_ttest": {
      "cohens_d": -9.058444273065838,
      "n": 45.0,
      "p": 4.3848201831002818e-44,
      "t": -60.765891494906654
    },
    "main_vs_shuffled_paired_ttest": {
      "cohens_d": -0.029361370820591093,
      "n": 45.0,
      "p": 0.8447642333329782,
      "t": -0.1969620632022614
    },
    "main_vs_wronglayer_paired_ttest": {
      "cohens_d": -1.2963884828980679,
      "n": 45.0,
      "p": 4.126844351048041e-11,
      "t": -8.696438319023708
    }
  },
  "timestamp": "20260124_100057",
  "transfer_percent_estimate": 94.22678484895333
}
```
