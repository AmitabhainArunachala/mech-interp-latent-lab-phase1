# Run report: behavioral_grounding

- **run_dir**: `results/phase1_mechanism/runs/20251213_103333_behavioral_grounding_mistral_base_behavioral_grounding_v1`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "csv": "results/phase1_mechanism/runs/20251213_103333_behavioral_grounding_mistral_base_behavioral_grounding_v1/behavioral_grounding_summary.csv",
    "jsonl": "results/phase1_mechanism/runs/20251213_103333_behavioral_grounding_mistral_base_behavioral_grounding_v1/behavioral_grounding.jsonl"
  },
  "by_condition": {
    "baseline": {
      "gen_token_count_mean": 160.0,
      "n": 8,
      "repeat_4gram_frac_mean": 0.5061464661379698,
      "self_ref_rate_mean": 0.0,
      "unique_word_ratio_mean": 0.4123399670230864
    },
    "baseline_patched": {
      "gen_token_count_mean": 160.0,
      "n": 8,
      "repeat_4gram_frac_mean": 0.25,
      "self_ref_rate_mean": 0.0,
      "unique_word_ratio_mean": 0.0019386574074074074
    },
    "recursive": {
      "gen_token_count_mean": 160.0,
      "n": 8,
      "repeat_4gram_frac_mean": 0.9818315365190364,
      "self_ref_rate_mean": 0.13372337483967803,
      "unique_word_ratio_mean": 0.14252429379835996
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
