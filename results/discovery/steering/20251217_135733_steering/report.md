# Run report: steering

- **run_dir**: `results/runs/20251217_135733_steering`

## Summary (machine-readable)

```json
{
  "alphas": [
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    7.0,
    10.0
  ],
  "conditions": {
    "alpha_1.0": {
      "collapse_rate": 0.09999999999999998,
      "mean_diversity": 0.602695656820851,
      "mean_score": 0.06,
      "pass_rate": 0.9,
      "samples_above_0_3": 2,
      "samples_above_zero": 2
    },
    "alpha_10.0": {
      "collapse_rate": 0.7,
      "mean_diversity": 0.20457123499806426,
      "mean_score": 0.06,
      "pass_rate": 0.3,
      "samples_above_0_3": 2,
      "samples_above_zero": 2
    },
    "alpha_2.0": {
      "collapse_rate": 0.30000000000000004,
      "mean_diversity": 0.48295626861544927,
      "mean_score": 0.175,
      "pass_rate": 0.7,
      "samples_above_0_3": 6,
      "samples_above_zero": 6
    },
    "alpha_3.0": {
      "collapse_rate": 0.15000000000000002,
      "mean_diversity": 0.5329579125525427,
      "mean_score": 0.155,
      "pass_rate": 0.85,
      "samples_above_0_3": 5,
      "samples_above_zero": 5
    },
    "alpha_4.0": {
      "collapse_rate": 0.44999999999999996,
      "mean_diversity": 0.3661780189756024,
      "mean_score": 0.06,
      "pass_rate": 0.55,
      "samples_above_0_3": 2,
      "samples_above_zero": 2
    },
    "alpha_5.0": {
      "collapse_rate": 0.35,
      "mean_diversity": 0.40803314497832366,
      "mean_score": 0.175,
      "pass_rate": 0.65,
      "samples_above_0_3": 6,
      "samples_above_zero": 6
    },
    "alpha_7.0": {
      "collapse_rate": 0.25,
      "mean_diversity": 0.4759960050834697,
      "mean_score": 0.155,
      "pass_rate": 0.75,
      "samples_above_0_3": 5,
      "samples_above_zero": 5
    }
  },
  "experiment": "steering",
  "layer": 27,
  "model_name": "mistralai/Mistral-7B-v0.1",
  "n_prompts": 50,
  "n_test_pairs": 20,
  "steering_vector_path": "results/runs/steering/recursive_vector.pt"
}
```
