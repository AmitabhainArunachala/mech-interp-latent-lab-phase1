# Run report: behavior_strict

- **run_dir**: `results/runs/20251217_122855_behavior_strict`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "csv": "results/runs/20251217_122855_behavior_strict/behavior_strict_results.csv"
  },
  "conditions": {
    "Baseline_Control": {
      "diversity": 0.5610738473628072,
      "mean_score": 0.145,
      "pass_rate": 0.8
    },
    "Random_Control": {
      "diversity": 0.7910513718471823,
      "mean_score": 0.06,
      "pass_rate": 0.95
    },
    "Recursive_Control": {
      "diversity": 0.3268960532931121,
      "mean_score": 0.09,
      "pass_rate": 0.45
    },
    "Shuffled_Control": {
      "diversity": 0.3863361667354409,
      "mean_score": 0.485,
      "pass_rate": 0.65
    },
    "Transfer": {
      "diversity": 0.1794025717452825,
      "mean_score": 0.17,
      "pass_rate": 0.3
    },
    "Transfer_L27_Only": {
      "diversity": 0.10769230769230768,
      "mean_score": 0.0,
      "pass_rate": 0.15
    }
  },
  "experiment": "behavior_strict",
  "model_name": "mistralai/Mistral-7B-v0.1",
  "n_pairs": 20,
  "prompt_bank_version": "b1e5291421c5646d"
}
```
