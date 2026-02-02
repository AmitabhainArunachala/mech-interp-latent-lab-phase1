# Run report: rv_l27_causal_validation

- **run_dir**: `results/phase2_generalization/gemma_2_9b/11_causal_validation_champion/runs/20260124_102546_rv_l27_causal_validation_gemma_2_9b_causal_validation_champion`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "config": "results/phase2_generalization/gemma_2_9b/11_causal_validation_champion/runs/20260124_102546_rv_l27_causal_validation_gemma_2_9b_causal_validation_champion/config.json",
    "pairs_csv": "results/phase2_generalization/gemma_2_9b/11_causal_validation_champion/runs/20260124_102546_rv_l27_causal_validation_gemma_2_9b_causal_validation_champion/rv_l27_causal_validation_pairs.csv",
    "report": "results/phase2_generalization/gemma_2_9b/11_causal_validation_champion/runs/20260124_102546_rv_l27_causal_validation_gemma_2_9b_causal_validation_champion/report.md",
    "summary": "results/phase2_generalization/gemma_2_9b/11_causal_validation_champion/runs/20260124_102546_rv_l27_causal_validation_gemma_2_9b_causal_validation_champion/summary.json"
  },
  "delta_main": {
    "ci_95": [
      -0.19247256911465516,
      -0.15007298062994928
    ],
    "mean": -0.17127277487230222,
    "n": 60.0,
    "std": 0.08206560689020734
  },
  "delta_random": {
    "ci_95": [
      1.1780456731670808,
      1.2784343512558562
    ],
    "mean": 1.2282400122114685,
    "n": 60.0,
    "std": 0.1943051356555677
  },
  "delta_shuffled": {
    "ci_95": [
      -0.1924725691146556,
      -0.15007298062994961
    ],
    "mean": -0.1712727748723026,
    "n": 60.0,
    "std": 0.0820656068902076
  },
  "delta_wronglayer": {
    "ci_95": [
      -0.04896336856826685,
      0.005893467810769541
    ],
    "mean": -0.021534950378748655,
    "n": 60.0,
    "std": 0.10617696374921921
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
        "champion",
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
      0.7506975292567827,
      0.7792736567729001
    ],
    "mean": 0.7649855930148414,
    "n": 60.0,
    "std": 0.05530990585033747
  },
  "rv_baseline_mean": 0.7649855930148414,
  "rv_cohens_d": -2.087022583058966,
  "rv_delta_mean": -0.17127277487230222,
  "rv_p_value": 1.204665032918014e-23,
  "rv_recursive": {
    "ci_95": [
      0.5807478007722449,
      0.6051136683948204
    ],
    "mean": 0.5929307345835326,
    "n": 60.0,
    "std": 0.047160828331491864
  },
  "rv_recursive_mean": 0.5929307345835326,
  "schema_version": "metrics_summary_v1",
  "seed": 42,
  "tests": {
    "main_effect_ttest_1samp_less_0": {
      "cohens_d": -2.087022583058966,
      "n": 60,
      "p": 1.204665032918014e-23,
      "t": -16.166007414692324
    },
    "main_vs_random_paired_ttest": {
      "cohens_d": -8.724379801925599,
      "n": 60.0,
      "p": 1.3622203465653355e-57,
      "t": -67.57875535769242
    },
    "main_vs_shuffled_paired_ttest": {
      "cohens_d": 0.0843194705300909,
      "n": 60.0,
      "p": 0.5162059299863186,
      "t": 0.6531358102481383
    },
    "main_vs_wronglayer_paired_ttest": {
      "cohens_d": -1.354566225705877,
      "n": 60.0,
      "p": 4.1181449000127995e-15,
      "t": -10.492424866987797
    }
  },
  "timestamp": "20260124_102611",
  "transfer_percent_estimate": 99.54544523407418
}
```
