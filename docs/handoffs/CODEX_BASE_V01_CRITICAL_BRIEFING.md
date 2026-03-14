# CRITICAL BRIEFING: Base v0.1 vs Instruct v0.2 Contamination

**Date**: 2026-03-12
**From**: Claude Opus 4.6 (5-agent audit, 1,969 files catalogued)
**For**: Any agent working on Mistral hardening or paper prep

## The Core Problem

The paper says "Mistral-7B-v0.1" (base) but **5 sections use Instruct-v0.2 data**. A 5-agent deep audit confirmed:

- **Instruct v0.2**: 15 experiment types, 95% paper-ready
- **Base v0.1**: 8 experiment types, 20% paper-ready
- **P0 Canonical (THE headline number)**: 4 Instruct runs, ZERO base runs

## What IS Already Base v0.1

These existing results are genuinely base v0.1 — do NOT re-run:
- `sustained_gnani_v3/comparison_summary.json` (behavioral sessions)
- `self_feeding_loop/self_feeding_summary_20260227_054825.json`
- `mode_atlas/atlas_summary_20260227_075328.json` (older, but real base)
- `power_up/mistral-7b_n80_result.json` (d=-1.656, n=75/77)
- `full_head_sweep/full_head_sweep_20260302_074757.json` (base, n=20, 606 entropy-sig)
- `persistent_patching_v3/persistent_patching_v3_dual_20260225_002604.json` (base, old schema)
- Safety, scaling_gap, scaling_law results

## What Is Instruct v0.2 (NOT base)

These are Instruct-v0.2 and the paper cannot cite them as "base":
- `p0_canonical/mistralai__Mistral-7B-Instruct-v0-2_p0_result.json` (g=-1.47)
- `full_head_sweep/full_head_sweep_20260310_151508.json` (630 entropy-sig, L22H21)
- `full_head_sweep/full_head_sweep_20260311_120236.json`
- `path_patching/path_patching_summary_20260310_151654.json` (all 32 layers)
- `path_patching/path_patching_summary_20260311_121417.json`
- ALL `persistent_patching_v3_dual_20260310_*` files (including the full rerun 204100)
- `circuit_mapping/` (all files)
- `gnani_protocol/` (all files)
- `mediation/mediation_2x2_20260311_121447.json`
- `svd_circuits/svd_decomposition_20260310_*.json`

## The 7 Core Base v0.1 Jobs Needed

To bring base v0.1 to paper parity, run these on base `mistralai/Mistral-7B-v0.1`:

1. **P0 Canonical** (2 GPU hrs) — `python scripts/p0_canonical_pipeline.py --model mistralai/Mistral-7B-v0.1`
2. **Full Head Sweep** n>=100 (16 GPU hrs)
3. **SVD Circuit Decomposition** n>=100
4. **Full 32-Layer Path Patching** (8 GPU hrs) — base currently only has EVEN layers
5. **Refreshed Persistent Dual Patch** (6 GPU hrs) — use current canonical pipeline
6. **Mediation 2x2** n>=100
7. **Mode Atlas refresh** on frozen `mistral_hardening_v1` subset

## 8 Paper Claims With No/Wrong Source Files

These are in the paper but CANNOT be traced to any result JSON:

| # | Claim | Paper Value | Reality | Action |
|---|-------|-------------|---------|--------|
| 1 | Per-token R_V bridge | d=-1.64, p=1.4e-6 | Closest real: d=-0.608 | **REMOVE** (fabricated/lost) |
| 2 | Primary headline | d=-2.26, n=151/151 | No primary file produces this | **Trace or replace with P0 canonical** |
| 3 | Gemma cross-arch | d=-3.37 | Raw data: d=-2.09 or -1.74 | **Fix number** |
| 4 | Mixtral | 24.3%, d~5.3 | NO result data exists | **Remove** |
| 5 | Bayes Factor | BF10=9.5e23 | Not stored in any file | **Recompute & store** |
| 6 | Head count | 630/691 | Computable but not stored | **Recompute from raw** |
| 7 | Word count | r=-0.171, p=0.498 | No source file | **Remove or run analysis** |
| 8 | Necessity h=1.31 | Cohen's h=1.31 | Source stores d=1.664 | **Clarify conversion** |

## Statistical Honesty Gap

- **FDR-corrected**: 39 tests (32 pass at q<0.05)
- **Uncorrected with p-values**: ~600 more tests
- **Total comparisons computed**: ~1,300-1,600
- **Paper needs**: transparency statement distinguishing confirmatory (39) from exploratory (~1,300)

## Decision Needed

**Option A**: Run base v0.1 as canonical (38 GPU hrs, 6-8 days)
**Option B**: Pivot to Instruct v0.2 as primary, acknowledge in methods
**Option C**: Hybrid — run just P0 canonical on base (2 hrs), keep Instruct for mechanism details

## What Codex Should NOT Do

- Do NOT run more Instruct v0.2 experiments expecting them to fill base gaps
- Do NOT update paper numbers from Instruct results while paper claims base
- Do NOT treat `persistent_patching_v3_dual_20260310_204100.json` as base (it's Instruct)

## What Codex SHOULD Do

- Run the base v0.1 queue (7 jobs above) on RunPod
- Use identical scripts, prompts (`mistral_hardening_v1`), and canonical registry
- Save results with explicit model provenance in filenames
- After base results land, THEN update paper numbers from the correct model
