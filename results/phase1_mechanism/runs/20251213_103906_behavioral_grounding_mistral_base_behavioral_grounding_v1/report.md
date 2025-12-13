# Run report: behavioral_grounding

- **run_dir**: `results/phase1_mechanism/runs/20251213_103906_behavioral_grounding_mistral_base_behavioral_grounding_v1`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "csv": "results/phase1_mechanism/runs/20251213_103906_behavioral_grounding_mistral_base_behavioral_grounding_v1/behavioral_grounding_summary.csv",
    "jsonl": "results/phase1_mechanism/runs/20251213_103906_behavioral_grounding_mistral_base_behavioral_grounding_v1/behavioral_grounding.jsonl"
  },
  "by_condition": {
    "baseline": {
      "gen_token_count_mean": 160.0,
      "n": 8,
      "repeat_4gram_frac_mean": 0.125,
      "self_ref_rate_mean": 0.0,
      "unique_word_ratio_mean": 0.13636363636363635
    },
    "baseline_patched": {
      "gen_token_count_mean": 160.0,
      "n": 8,
      "repeat_4gram_frac_mean": 0.17708333333333331,
      "self_ref_rate_mean": 0.0,
      "unique_word_ratio_mean": 0.11342592592592593
    },
    "recursive": {
      "gen_token_count_mean": 160.0,
      "n": 8,
      "repeat_4gram_frac_mean": 0.09375,
      "self_ref_rate_mean": 0.0,
      "unique_word_ratio_mean": 0.018333333333333333
    }
  },
  "device": "cuda",
  "experiment": "behavioral_grounding",
  "model_name": "mistralai/Mistral-7B-v0.1",
  "params": {
    "do_sample": false,
    "max_new_tokens": 160,
    "max_pairs": 8,
    "pairing": {
      "baseline_groups": [
        "long_control",
        "baseline_creative",
        "baseline_math"
      ],
      "recursive_groups": [
        "L5_refined",
        "L4_full",
        "L3_deeper"
      ]
    },
    "patch_layer": 24,
    "temperature": 0.8,
    "window": 16
  }
}
```
