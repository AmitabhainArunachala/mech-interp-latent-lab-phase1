# E2.2 Full Head Sweep — Run Log

**Command**: `python3 scripts/full_head_sweep.py --device cuda --n-prompts 20 --batch-layers 4`
**Model**: mistralai/Mistral-7B-v0.1
**GPU**: RunPod A100-80GB (inferred from remote_gpu_sync timestamps)
**Date**: 2026-03-02 ~07:47 UTC
**Runtime**: Estimated ~45-60 min for 32 layers × 32 heads

## Results
- 1024/1024 heads measured (complete)
- 606/1024 significant at p<0.05 for entropy divergence
- 256/1024 have valid rank_d (remaining 768 have NaN rank — likely OV extraction issue for non-GQA layers)
- Mean entropy Cohen's d = 0.508 (median = 0.550)
- Top discriminating head: L10H20 (d=3.90)

## Errors
- 768 heads have NaN for rank_d/rank_p: OV effective rank computation failed for layers 0-7 (heads 8-31 per layer). Likely caused by `capture_v_projection` hook not capturing for grouped-query attention heads. Entropy metrics are complete for all heads.

## Artifacts
- Raw: `results/full_head_sweep/full_head_sweep_20260302_074757.json`
- Metrics: `results/rv_masterplan/E2.2_full_head_sweep/metrics.json`
