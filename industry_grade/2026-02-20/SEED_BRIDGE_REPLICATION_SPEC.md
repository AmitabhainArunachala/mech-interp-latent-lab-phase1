# Seed Bridge Replication Spec (Pre-Registered) — 2026-02-20

## Objective
Replicate and stress-test the bridge specificity finding on Mistral-7B with independent seeds and fixed controls.

Primary claim to test:
- Head-specific recursive donor intervention decreases `rv_delta_mean` relative to both control interventions.

## Design

Conditions (fixed):
1. `head_specific`
2. `random_head_control`
3. `baseline_donor_control`

Model and geometry contract:
- Model: `mistralai/Mistral-7B-v0.1`
- Early layer: `5`
- Target layer: `27`
- Window: `16`
- Prompt bank source: `prompts/bank.json` via `PromptLoader`

Run tier:
- Tier 1 deterministic (primary): `temperature=0.0`, `do_sample=false`

## Replication Matrix

Seeds:
- `42, 123, 456, 789, 1024` (5 independent seeds)

Per-run sample size:
- `n_pairs=80` (publication-grade target from README standards)

Total runs:
- `3 conditions x 5 seeds = 15 runs`
- Config matrix: `configs/canonical/seed_bridge_2026_02_20/` (`RUN_MATRIX.csv`)
- Batch launcher: `industry_grade/2026-02-20/run_seed_bridge_matrix.sh`
- Analyzer: `industry_grade/2026-02-20/analyze_seed_bridge_matrix.py`

## Required Artifacts Per Run

- `config.json`
- `summary.json`
- `report.md`
- `prompt_bank_version.json`
- `metadata.json`
- `hardware_info.json`
- `per_sample.csv`

All runs must be executed via:
- `python -m src.pipelines.run --config ...`

## Required Summary Keys

Per `summary.json`:
- `n_pairs`
- `rv_delta_mean`
- `rv_cohens_d`
- `rv_p_value`
- `prompt_bank_version`

## Analysis Plan

Within-seed contrasts (paired tests over matched `(rec_id, base_id)` per-sample `rv_delta`; Welch fallback if alignment missing):
- `head_specific` vs `random_head_control`
- `head_specific` vs `baseline_donor_control`
- `random_head_control` vs `baseline_donor_control`

Across-seed aggregation:
- Fixed-effects and random-effects pooled mean difference for each contrast
- Report 95% CI and heterogeneity (`I2`)

## Pass/Fail Criteria (Pre-registered)

Primary pass:
1. For at least `4/5` seeds:
- `head_specific - random_head_control < 0`
- `head_specific - baseline_donor_control < 0`

2. Pooled random-effects estimate for both primary contrasts is `< 0` with 95% CI excluding `0`.

Secondary sanity:
- `random_head_control` vs `baseline_donor_control` not directionally consistent across seeds OR pooled CI includes `0`.

Minimum effect-size expectation:
- Pooled absolute Cohen's `|d| >= 0.5` for each primary contrast.

## Operational Guards

- Abort if artifact contract fails for any run.
- Abort if prompt bank version drifts within the run set.
- Record GPU/hardware metadata for every run.
- Use explicit runtime safeguards in matrix configs:
  - `generation_timeout_sec=120`
  - `checkpoint_every=1`

## Pre-Launch Gate (must remain true)

- `industry_grade/2026-02-20/evidence/verify_research_ready.txt` shows `RESULT: RESEARCH READY`
- `industry_grade/2026-02-20/evidence/artifact_contract_audit.json` shows `"all_pass": true`

If either fails, do not launch the 15-run seed matrix.
