# Run report: confound_validation

- **run_dir**: `results/gold_standard/runs/20251216_060911_confound_validation`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "confound_results_csv": "results/gold_standard/runs/20251216_060911_confound_validation/confound_results.csv"
  },
  "corr_token_count_vs_rv": {
    "p": 0.08579296084461316,
    "r": 0.28632070216663164
  },
  "mean_rv": {
    "champions": 0.5185336374047422,
    "length_matched": 0.8322957867208634,
    "pseudo_recursive": 0.7791983027949027
  },
  "n_champions": 15,
  "n_length_matched": 11,
  "n_pseudo_recursive": 11,
  "n_total": 37,
  "params": {
    "early_layer": 5,
    "late_layer": 27,
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "window": 16
  },
  "ttest": {
    "champions_vs_length_matched": {
      "p": 4.276013273805805e-05,
      "t": -6.581271038675067
    },
    "champions_vs_pseudo_recursive": {
      "p": 2.1571287365221866e-06,
      "t": -8.408000717891673
    },
    "length_matched_vs_pseudo_recursive": {
      "p": 0.35024134770193205,
      "t": 0.9607361336845651
    }
  }
}
```
