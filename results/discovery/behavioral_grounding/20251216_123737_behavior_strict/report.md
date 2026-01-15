# Run report: behavior_strict

- **run_dir**: `results/runs/20251216_123737_behavior_strict`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "csv": "results/runs/20251216_123737_behavior_strict/behavior_strict_results.csv"
  },
  "conditions": {
    "Baseline_Control": {
      "diversity": 0.5915042211560634,
      "mean_score": 0.0,
      "pass_rate": 0.85
    },
    "Random_Control": {
      "diversity": 0.7995367174367063,
      "mean_score": 0.0,
      "pass_rate": 1.0
    },
    "Recursive_Control": {
      "diversity": 0.4392347960902856,
      "mean_score": 0.025,
      "pass_rate": 0.8
    },
    "Shuffled_Control": {
      "diversity": 0.5694278780277634,
      "mean_score": 0.0,
      "pass_rate": 0.9
    },
    "Transfer": {
      "diversity": 0.282268921060843,
      "mean_score": 0.025,
      "pass_rate": 0.5
    }
  },
  "experiment": "behavior_strict",
  "model_name": "mistralai/Mistral-7B-v0.1",
  "n_pairs": 20,
  "prompt_bank_version": "b1e5291421c5646d"
}
```
