# Run report: rv_l27_causal_validation

- **run_dir**: `results/phase2_generalization/gemma_2_9b/11_causal_validation_champion/runs/20260124_112226_rv_l27_causal_validation_gemma_2_9b_causal_validation_champion`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "config": "results/phase2_generalization/gemma_2_9b/11_causal_validation_champion/runs/20260124_112226_rv_l27_causal_validation_gemma_2_9b_causal_validation_champion/config.json",
    "pairs_csv": "results/phase2_generalization/gemma_2_9b/11_causal_validation_champion/runs/20260124_112226_rv_l27_causal_validation_gemma_2_9b_causal_validation_champion/rv_l27_causal_validation_pairs.csv",
    "report": "results/phase2_generalization/gemma_2_9b/11_causal_validation_champion/runs/20260124_112226_rv_l27_causal_validation_gemma_2_9b_causal_validation_champion/report.md",
    "summary": "results/phase2_generalization/gemma_2_9b/11_causal_validation_champion/runs/20260124_112226_rv_l27_causal_validation_gemma_2_9b_causal_validation_champion/summary.json"
  },
  "delta_main": {
    "ci_95": [
      -0.20398801955979362,
      -0.1511324426831519
    ],
    "mean": -0.17756023112147276,
    "n": 60.0,
    "std": 0.10230346918292048
  },
  "delta_random": {
    "ci_95": [
      1.235163403023805,
      1.3439910500819803
    ],
    "mean": 1.2895772265528926,
    "n": 60.0,
    "std": 0.21063899960925286
  },
  "delta_shuffled": {
    "ci_95": [
      -0.20398801955979445,
      -0.15113244268315307
    ],
    "mean": -0.17756023112147376,
    "n": 60.0,
    "std": 0.10230346918291981
  },
  "delta_wronglayer": {
    "ci_95": [
      -0.042316211662375244,
      0.015093720864011142
    ],
    "mean": -0.01361124539918205,
    "n": 60.0,
    "std": 0.11111855380396479
  },
  "device": "cuda",
  "experiment": "rv_l27_causal_validation",
  "logit_diff_cohens_d": null,
  "logit_diff_delta_mean": null,
  "logit_diff_p_value": null,
  "model": "google/gemma-2-9b",
  "model_name": "google/gemma-2-9b",
  "n_pairs": 60,
  "notes": {
    "measurement": "main/random/shuffled measured at target_layer; wronglayer measured at wrong_layer"
  },
  "params": {
    "early_layer": 5,
    "max_length": 512,
    "max_pairs": 60,
    "measure_target_after_wrong_patch": false,
    "pairing": {
      "baseline_groups": [
        "long_control",
        "baseline_creative",
        "baseline_math",
        "baseline_factual"
      ],
      "recursive_groups": [
        "champions",
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
      0.7555370956226051,
      0.7813212852650151
    ],
    "mean": 0.7684291904438101,
    "n": 60.0,
    "std": 0.049906030855461095
  },
  "rv_baseline_mean": 0.7684291904438101,
  "rv_cohens_d": -1.7356227754504767,
  "rv_delta_mean": -0.17756023112147276,
  "rv_p_value": 6.461161407866468e-20,
  "rv_recursive": {
    "ci_95": [
      0.5801102899308856,
      0.6059794437644772
    ],
    "mean": 0.5930448668476814,
    "n": 60.0,
    "std": 0.05007048145893251
  },
  "rv_recursive_mean": 0.5930448668476814,
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "tests": {
    "main_effect_ttest_1samp_less_0": {
      "cohens_d": -1.7356227754504767,
      "n": 60,
      "p": 6.461161407866468e-20,
      "t": -13.444076209235984
    },
    "main_vs_random_paired_ttest": {
      "cohens_d": -9.251338984933568,
      "n": 60.0,
      "p": 4.459813444141553e-59,
      "t": -71.66056363753428
    },
    "main_vs_shuffled_paired_ttest": {
      "cohens_d": 0.23704219210114244,
      "n": 60.0,
      "p": 0.07137986990803719,
      "t": 1.836120924712448
    },
    "main_vs_wronglayer_paired_ttest": {
      "cohens_d": -1.7497006105197985,
      "n": 60.0,
      "p": 8.998177476538991e-20,
      "t": -13.553122650784259
    }
  },
  "timestamp": "20260124_112250",
  "transfer_percent_estimate": 101.24065109168745
}
```
