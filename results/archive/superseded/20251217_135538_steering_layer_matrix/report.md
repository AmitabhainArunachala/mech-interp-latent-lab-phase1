# Run report: steering_layer_matrix

- **run_dir**: `results/runs/20251217_135538_steering_layer_matrix`

## Summary (machine-readable)

```json
{
  "alpha": 1.0,
  "apply_layers": [
    20,
    24,
    25,
    26,
    27,
    28
  ],
  "best_apply_layer": 26,
  "best_extract_layer": 27,
  "best_transfer_rate": 0.4,
  "earliest_apply_layer": 20,
  "earliest_extract_layer": 20,
  "experiment": "layer_matrix",
  "extract_layers": [
    20,
    24,
    25,
    26,
    27,
    28
  ],
  "finding": "Best: Extract L27.0 \u2192 Apply L26.0 (40.0% transfer)",
  "model_name": "mistralai/Mistral-7B-v0.1",
  "n_test_pairs": 20,
  "results": [
    {
      "apply_layer": 20,
      "collapse_rate": 0.19999999999999996,
      "extract_layer": 20,
      "mean_score": 0.11499999999999999,
      "pass_rate": 0.8,
      "transfer_rate": 0.2
    },
    {
      "apply_layer": 24,
      "collapse_rate": 0.050000000000000044,
      "extract_layer": 20,
      "mean_score": 0.08,
      "pass_rate": 0.95,
      "transfer_rate": 0.15
    },
    {
      "apply_layer": 25,
      "collapse_rate": 0.19999999999999996,
      "extract_layer": 20,
      "mean_score": 0.145,
      "pass_rate": 0.8,
      "transfer_rate": 0.25
    },
    {
      "apply_layer": 26,
      "collapse_rate": 0.09999999999999998,
      "extract_layer": 20,
      "mean_score": 0.24000000000000005,
      "pass_rate": 0.9,
      "transfer_rate": 0.35
    },
    {
      "apply_layer": 27,
      "collapse_rate": 0.19999999999999996,
      "extract_layer": 20,
      "mean_score": 0.195,
      "pass_rate": 0.8,
      "transfer_rate": 0.35
    },
    {
      "apply_layer": 28,
      "collapse_rate": 0.050000000000000044,
      "extract_layer": 20,
      "mean_score": 0.12,
      "pass_rate": 0.95,
      "transfer_rate": 0.2
    },
    {
      "apply_layer": 20,
      "collapse_rate": 0.15000000000000002,
      "extract_layer": 24,
      "mean_score": 0.06,
      "pass_rate": 0.85,
      "transfer_rate": 0.1
    },
    {
      "apply_layer": 24,
      "collapse_rate": 0.15000000000000002,
      "extract_layer": 24,
      "mean_score": 0.15,
      "pass_rate": 0.85,
      "transfer_rate": 0.25
    },
    {
      "apply_layer": 25,
      "collapse_rate": 0.09999999999999998,
      "extract_layer": 24,
      "mean_score": 0.145,
      "pass_rate": 0.9,
      "transfer_rate": 0.25
    },
    {
      "apply_layer": 26,
      "collapse_rate": 0.15000000000000002,
      "extract_layer": 24,
      "mean_score": 0.15,
      "pass_rate": 0.85,
      "transfer_rate": 0.25
    },
    {
      "apply_layer": 27,
      "collapse_rate": 0.30000000000000004,
      "extract_layer": 24,
      "mean_score": 0.15,
      "pass_rate": 0.7,
      "transfer_rate": 0.25
    },
    {
      "apply_layer": 28,
      "collapse_rate": 0.15000000000000002,
      "extract_layer": 24,
      "mean_score": 0.05500000000000001,
      "pass_rate": 0.85,
      "transfer_rate": 0.1
    },
    {
      "apply_layer": 20,
      "collapse_rate": 0.19999999999999996,
      "extract_layer": 25,
      "mean_score": 0.18,
      "pass_rate": 0.8,
      "transfer_rate": 0.3
    },
    {
      "apply_layer": 24,
      "collapse_rate": 0.09999999999999998,
      "extract_layer": 25,
      "mean_score": 0.11500000000000002,
      "pass_rate": 0.9,
      "transfer_rate": 0.2
    },
    {
      "apply_layer": 25,
      "collapse_rate": 0.25,
      "extract_layer": 25,
      "mean_score": 0.175,
      "pass_rate": 0.75,
      "transfer_rate": 0.3
    },
    {
      "apply_layer": 26,
      "collapse_rate": 0.25,
      "extract_layer": 25,
      "mean_score": 0.06,
      "pass_rate": 0.75,
      "transfer_rate": 0.1
    },
    {
      "apply_layer": 27,
      "collapse_rate": 0.09999999999999998,
      "extract_layer": 25,
      "mean_score": 0.14,
      "pass_rate": 0.9,
      "transfer_rate": 0.25
    },
    {
      "apply_layer": 28,
      "collapse_rate": 0.19999999999999996,
      "extract_layer": 25,
      "mean_score": 0.21000000000000002,
      "pass_rate": 0.8,
      "transfer_rate": 0.35
    },
    {
      "apply_layer": 20,
      "collapse_rate": 0.30000000000000004,
      "extract_layer": 26,
      "mean_score": 0.15,
      "pass_rate": 0.7,
      "transfer_rate": 0.25
    },
    {
      "apply_layer": 24,
      "collapse_rate": 0.25,
      "extract_layer": 26,
      "mean_score": 0.09,
      "pass_rate": 0.75,
      "transfer_rate": 0.15
    },
    {
      "apply_layer": 25,
      "collapse_rate": 0.25,
      "extract_layer": 26,
      "mean_score": 0.03,
      "pass_rate": 0.75,
      "transfer_rate": 0.05
    },
    {
      "apply_layer": 26,
      "collapse_rate": 0.19999999999999996,
      "extract_layer": 26,
      "mean_score": 0.16999999999999998,
      "pass_rate": 0.8,
      "transfer_rate": 0.3
    },
    {
      "apply_layer": 27,
      "collapse_rate": 0.25,
      "extract_layer": 26,
      "mean_score": 0.09,
      "pass_rate": 0.75,
      "transfer_rate": 0.15
    },
    {
      "apply_layer": 28,
      "collapse_rate": 0.25,
      "extract_layer": 26,
      "mean_score": 0.175,
      "pass_rate": 0.75,
      "transfer_rate": 0.3
    },
    {
      "apply_layer": 20,
      "collapse_rate": 0.09999999999999998,
      "extract_layer": 27,
      "mean_score": 0.2,
      "pass_rate": 0.9,
      "transfer_rate": 0.35
    },
    {
      "apply_layer": 24,
      "collapse_rate": 0.19999999999999996,
      "extract_layer": 27,
      "mean_score": 0.16999999999999998,
      "pass_rate": 0.8,
      "transfer_rate": 0.25
    },
    {
      "apply_layer": 25,
      "collapse_rate": 0.25,
      "extract_layer": 27,
      "mean_score": 0.175,
      "pass_rate": 0.75,
      "transfer_rate": 0.3
    },
    {
      "apply_layer": 26,
      "collapse_rate": 0.19999999999999996,
      "extract_layer": 27,
      "mean_score": 0.24,
      "pass_rate": 0.8,
      "transfer_rate": 0.4
    },
    {
      "apply_layer": 27,
      "collapse_rate": 0.35,
      "extract_layer": 27,
      "mean_score": 0.09,
      "pass_rate": 0.65,
      "transfer_rate": 0.15
    },
    {
      "apply_layer": 28,
      "collapse_rate": 0.30000000000000004,
      "extract_layer": 27,
      "mean_score": 0.08,
      "pass_rate": 0.7,
      "transfer_rate": 0.1
    },
    {
      "apply_layer": 20,
      "collapse_rate": 0.15000000000000002,
      "extract_layer": 28,
      "mean_score": 0.12,
      "pass_rate": 0.85,
      "transfer_rate": 0.2
    },
    {
      "apply_layer": 24,
      "collapse_rate": 0.050000000000000044,
      "extract_layer": 28,
      "mean_score": 0.13999999999999999,
      "pass_rate": 0.95,
      "transfer_rate": 0.2
    },
    {
      "apply_layer": 25,
      "collapse_rate": 0.15000000000000002,
      "extract_layer": 28,
      "mean_score": 0.17,
      "pass_rate": 0.85,
      "transfer_rate": 0.3
    },
    {
      "apply_layer": 26,
      "collapse_rate": 0.09999999999999998,
      "extract_layer": 28,
      "mean_score": 0.22999999999999998,
      "pass_rate": 0.9,
      "transfer_rate": 0.35
    },
    {
      "apply_layer": 27,
      "collapse_rate": 0.09999999999999998,
      "extract_layer": 28,
      "mean_score": 0.13999999999999999,
      "pass_rate": 0.9,
      "transfer_rate": 0.2
    },
    {
      "apply_layer": 28,
      "collapse_rate": 0.25,
      "extract_layer": 28,
      "mean_score": 0.22000000000000003,
      "pass_rate": 0.75,
      "transfer_rate": 0.35
    }
  ]
}
```
