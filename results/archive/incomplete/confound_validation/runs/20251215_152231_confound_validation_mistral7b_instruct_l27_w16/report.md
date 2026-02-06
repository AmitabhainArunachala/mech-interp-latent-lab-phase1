# Run report: confound_validation

- **run_dir**: `results/confound_validation/runs/20251215_152231_confound_validation_mistral7b_instruct_l27_w16`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "confound_results_csv": "results/confound_validation/runs/20251215_152231_confound_validation_mistral7b_instruct_l27_w16/confound_results.csv"
  },
  "corr_token_count_vs_rv": {
    "p": 0.0056983141503577145,
    "r": 0.37135475902019216
  },
  "mean_rv": {
    "champions": 0.4570926315170731,
    "length_matched": 0.7666414896902634,
    "pseudo_recursive": 0.7173693986897671
  },
  "n_champions": 18,
  "n_length_matched": 18,
  "n_pseudo_recursive": 18,
  "n_total": 54,
  "params": {
    "early_layer": 5,
    "late_layer": 27,
    "model_name": "mistralai/Mistral-7B-v0.1",
    "window": 16
  },
  "ttest": {
    "champions_vs_length_matched": {
      "p": 3.21703373576095e-07,
      "t": -7.933029661309594
    },
    "champions_vs_pseudo_recursive": {
      "p": 3.0925402090374855e-10,
      "t": -11.877876005509146
    },
    "length_matched_vs_pseudo_recursive": {
      "p": 0.2744766477471559,
      "t": 1.1159663676190776
    }
  }
}
```
