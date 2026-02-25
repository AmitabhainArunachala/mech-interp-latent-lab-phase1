# Run report: c2_rv_measurement

- **run_dir**: `results/runs/20260208_234928_c2_rv_measurement`
- **prompt_bank_version**: `75e7c1b8dcebc24e`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "config": "results/runs/20260208_234928_c2_rv_measurement/config.json",
    "csv": "results/runs/20260208_234928_c2_rv_measurement/c2_rv_measurement.csv",
    "outputs_dir": "results/runs/20260208_234928_c2_rv_measurement/outputs",
    "report": "results/runs/20260208_234928_c2_rv_measurement/report.md",
    "summary": "results/runs/20260208_234928_c2_rv_measurement/summary.json"
  },
  "by_config": {
    "baseline": {
      "coherence": 0.8699999999999999,
      "crystallization_layer_mean": 24.9,
      "logit_diff_mean": -4.67724609375,
      "logit_diff_std": 4.680538144756853,
      "mode_score_m_mean": null,
      "n_prompts": 20,
      "philosophical_pct": 0.0,
      "rv_ci_95_high": 0.7457477415567701,
      "rv_ci_95_low": 0.6547000741969566,
      "rv_mean": 0.7002239078768634,
      "rv_min": 0.5468345757405452,
      "rv_std": 0.09445081099273003,
      "task_pct": 45.0
    },
    "no_kv": {
      "coherence": 0.8300000000000001,
      "crystallization_layer_mean": 25.05,
      "logit_diff_mean": -5.88037109375,
      "logit_diff_std": 3.607129413384022,
      "mode_score_m_mean": null,
      "n_prompts": 20,
      "philosophical_pct": 0.0,
      "rv_ci_95_high": 0.6944300460409326,
      "rv_ci_95_low": 0.6383394774534047,
      "rv_mean": 0.6663847617471687,
      "rv_min": 0.504998060660223,
      "rv_std": 0.058187099634292076,
      "task_pct": 40.0
    }
  },
  "experiment": "c2_rv_measurement",
  "model": "mistralai/Mistral-7B-v0.1",
  "n_prompts": 20,
  "prompt_bank_version": "75e7c1b8dcebc24e",
  "schema_version": "metrics_summary_v1",
  "statistics": {},
  "timestamp": "20260208_235106"
}
```
