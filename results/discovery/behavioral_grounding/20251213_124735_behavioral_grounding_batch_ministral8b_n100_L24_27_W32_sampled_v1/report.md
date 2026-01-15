# Run report: behavioral_grounding_batch

- **run_dir**: `results/phase1_mechanism/runs/20251213_124735_behavioral_grounding_batch_ministral8b_n100_L24_27_W32_sampled_v1`

## Summary (machine-readable)

```json
{
  "artifacts": {
    "csv": "results/phase1_mechanism/runs/20251213_124735_behavioral_grounding_batch_ministral8b_n100_L24_27_W32_sampled_v1/behavioral_grounding_batch_summary.csv",
    "jsonl": "results/phase1_mechanism/runs/20251213_124735_behavioral_grounding_batch_ministral8b_n100_L24_27_W32_sampled_v1/behavioral_grounding_batch.jsonl"
  },
  "by_layer_condition": {
    "L24:baseline": {
      "gen_token_count_mean": 107.70769230769231,
      "repeat_4gram_frac_mean": 0.0957207478584819,
      "self_ref_rate_mean": 0.0016299168625204183,
      "unique_word_ratio_mean": 0.6075629793344193
    },
    "L24:baseline_patched": {
      "gen_token_count_mean": 88.29230769230769,
      "repeat_4gram_frac_mean": 0.5515757324280728,
      "self_ref_rate_mean": 0.014743589743589743,
      "unique_word_ratio_mean": 0.36236947431612165
    },
    "L25:baseline": {
      "gen_token_count_mean": 110.92307692307692,
      "repeat_4gram_frac_mean": 0.05817000296922009,
      "self_ref_rate_mean": 0.003390635799616275,
      "unique_word_ratio_mean": 0.5874045271250626
    },
    "L25:baseline_patched": {
      "gen_token_count_mean": 96.2,
      "repeat_4gram_frac_mean": 0.44520976219365416,
      "self_ref_rate_mean": 0.022705403511328458,
      "unique_word_ratio_mean": 0.4110055553099283
    },
    "L26:baseline": {
      "gen_token_count_mean": 107.66153846153846,
      "repeat_4gram_frac_mean": 0.07728648358023758,
      "self_ref_rate_mean": 0.0028807016920991443,
      "unique_word_ratio_mean": 0.5873824329362213
    },
    "L26:baseline_patched": {
      "gen_token_count_mean": 104.50769230769231,
      "repeat_4gram_frac_mean": 0.48327173132777745,
      "self_ref_rate_mean": 0.005128205128205128,
      "unique_word_ratio_mean": 0.40595397434474534
    },
    "L27:baseline": {
      "gen_token_count_mean": 109.92307692307692,
      "repeat_4gram_frac_mean": 0.08918109479929393,
      "self_ref_rate_mean": 0.0015170260495371048,
      "unique_word_ratio_mean": 0.5907070785001317
    },
    "L27:baseline_patched": {
      "gen_token_count_mean": 105.67692307692307,
      "repeat_4gram_frac_mean": 0.46600793651293115,
      "self_ref_rate_mean": 0.020865854575531993,
      "unique_word_ratio_mean": 0.4347968161470436
    }
  },
  "device": "cuda",
  "do_sample": true,
  "experiment": "behavioral_grounding_batch",
  "include_recursive_generation": false,
  "model_name": "mistralai/Ministral-8B-Instruct-2410",
  "n_pairs": 65,
  "patch_layers": [
    24,
    25,
    26,
    27
  ],
  "temperature": 0.7,
  "window": 32
}
```
