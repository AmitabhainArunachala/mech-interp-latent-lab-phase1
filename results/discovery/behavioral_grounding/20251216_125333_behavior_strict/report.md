# Run report: behavior_strict

- **run_dir**: `results/runs/20251216_125333_behavior_strict`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "csv": "results/runs/20251216_125333_behavior_strict/behavior_strict_results.csv"
  },
  "conditions": {
    "Baseline_Control": {
      "diversity": 0.5660325230428558,
      "mean_score": 0.0,
      "pass_rate": 0.8
    },
    "Random_Control": {
      "diversity": 0.7995367174367063,
      "mean_score": 0.0,
      "pass_rate": 1.0
    },
    "Recursive_Control": {
      "diversity": 0.41626182311731263,
      "mean_score": 0.315,
      "pass_rate": 0.75
    },
    "Shuffled_Control": {
      "diversity": 0.49980136892757016,
      "mean_score": 0.33999999999999997,
      "pass_rate": 0.8
    },
    "Transfer": {
      "diversity": 0.2605821740728912,
      "mean_score": 0.125,
      "pass_rate": 0.45
    }
  },
  "experiment": "behavior_strict",
  "model_name": "mistralai/Mistral-7B-v0.1",
  "n_pairs": 20,
  "prompt_bank_version": "b1e5291421c5646d"
}
```
