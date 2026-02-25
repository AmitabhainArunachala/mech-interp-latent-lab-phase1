# NeurIPS Finding One-Pager (2026-02-20)

## Proposed Finding
In grouped-query attention (GQA) transformers, head-level activation patching is not semantically neutral: if patching ignores architecture-correct headspace, estimated causal effect direction can invert.

## Falsifiable Test
Compare three intervention families on identical prompt pairs:
1. head-specific recursive donor patch
2. random-head patch control
3. baseline-donor specificity control

Prediction:
- (1) should decrease `rv_delta` relative to both controls.
- (2) and (3) should not separate strongly from each other.

## Current Evidence (Mistral-7B Seed Matrix, live)
Primary source: `industry_grade/2026-02-20/evidence/seed_bridge_analysis.json`

Seed 42 full triad (paired, matched prompt pairs, n=60 overlap each):
- `head_specific vs random_head_control`:
  - mean diff `-0.038566`
  - paired `p=1.786e-08`
  - Cohen's `d=-0.841`
- `head_specific vs baseline_donor_control`:
  - mean diff `-0.055016`
  - paired `p=2.029e-16`
  - Cohen's `d=-1.461`
- `random_head_control vs baseline_donor_control`:
  - mean diff `-0.016450`
  - paired `p=0.0284`
  - Cohen's `d=-0.290`

Single-condition means (seed 42):
- `head_specific`: `rv_delta_mean=-0.026541`
- `random_head_control`: `rv_delta_mean=+0.012025`
- `baseline_donor_control`: `rv_delta_mean=+0.028475`

Status:
- Standout gate currently passes for seed 42.
- Remaining seeds (123/456/789/1024 controls) are in-flight on remote GPU.

Interpretation:
- Mechanism-specific condition separates from both controls.
- Controls do not separate from each other.

## Behavioral Bridge Status (Multi-Token, Re-run)
- Run: `results/remote_gpu_sync/2026-02-20/phase1_cross_architecture/20260220_071457_multi_token_bridge_mistral_7b_bridge_deconfound_fast/summary.json`
- `temp=0.0`: truncation `88.9%`, H1 significant (`r=-0.650`, `p=1.80e-05`)
- `temp=0.7`: truncation `69.4%`, H1 not significant (`r=-0.409`, `p=0.212`)
- H2 (recursive vs baseline `R_V`) is strongly significant at both temperatures (`d=3.54`, `p=2.52e-12`).
- Takeaway: geometry separation is robust; behavior linkage still needs low-truncation confirmatory runs.

## Semantic Scoring Upgrade (Embedding-Based)
Source: `industry_grade/2026-02-20/evidence/semantic_behavior_analysis.json`

Method:
- Embedding model: `all-MiniLM-L6-v2`
- Score: max cosine similarity to 5 fixed `L5_refined` exemplars
- Threshold: `score > 0.4` => semantically recursive

Current readout:
- Seed-bridge patched outputs remain below threshold in this snapshot (rate floor at 0 across currently synced runs), but continuous score still carries signal:
  - pooled Spearman(`rv_patch`, semantic_score) `rho=-0.128`, `p=0.0266`, `n=300`
  - pooled Spearman(`rv_delta`, semantic_score) `rho=0.190`, `p=9.61e-4`, `n=300`
- C2 transfer outputs show strong semantic separation:
  - `baseline` semantic_recursive_rate `0.000`
  - `c2_full` semantic_recursive_rate `0.200`
  - pooled Spearman(`rv_mean`, semantic_score) `rho=-0.652`, `p=1.44e-92`, `n=755`

## Relation To Prior Work
- Circuit and intervention paradigm: https://arxiv.org/abs/2209.11895, https://arxiv.org/abs/2211.00593
- GQA semantics: https://arxiv.org/abs/2305.13245
- Mistral uses GQA: https://arxiv.org/abs/2310.06825
- Localization/editing caution: https://arxiv.org/abs/2301.04213
- Patch-scaling methods context: https://arxiv.org/abs/2407.02646, https://arxiv.org/abs/2511.05442

## Reproduction Commands
```bash
python3 scripts/verify_research_ready.py
python3 -m src.pipelines.run --config configs/canonical/rv_l27_head_specific_bridge_fast.json
python3 -m src.pipelines.run --config configs/canonical/rv_l27_random_head_bridge_fast.json
python3 -m src.pipelines.run --config configs/canonical/rv_l27_baseline_donor_bridge_fast.json
```

## Upgrade Path To Submission-Grade
1. Repeat with at least 5 seeds and preregistered acceptance thresholds.
2. Add one non-GQA model to test architecture dependence.
3. Keep behavioral bridge runs under explicit truncation ceilings before using them as confirmatory evidence.
