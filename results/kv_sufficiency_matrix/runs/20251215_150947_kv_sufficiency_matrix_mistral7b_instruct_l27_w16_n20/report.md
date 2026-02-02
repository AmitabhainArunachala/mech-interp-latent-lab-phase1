# Run report: kv_sufficiency_matrix

- **run_dir**: `results/kv_sufficiency_matrix/runs/20251215_150947_kv_sufficiency_matrix_mistral7b_instruct_l27_w16_n20`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "kv_matrix_results_csv": "results/kv_sufficiency_matrix/runs/20251215_150947_kv_sufficiency_matrix_mistral7b_instruct_l27_w16_n20/kv_matrix_results.csv"
  },
  "expression_rate_by_condition": {
    "A_control": 0.34,
    "B_kv_from_recursive": 0.56,
    "C_kv_from_baseline": 0.44,
    "D_random_kv_seed_101": 0.34,
    "D_random_kv_seed_202": 0.34,
    "D_random_kv_seed_303": 0.34,
    "E_vproj_only": 0.34
  },
  "n_pairs": 50,
  "n_rows": 350,
  "params": {
    "do_sample": true,
    "max_new_tokens": 100,
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "random_kv_seeds": [
      101,
      202,
      303
    ],
    "temperature": 0.7
  }
}
```
