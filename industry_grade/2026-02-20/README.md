# Industry-Grade Package (2026-02-20)

This folder is the dated handoff package requested for `mech-interp-latent-lab-phase1`.

## Purpose
- Lock a standards-compliant snapshot before launching multi-seed bridge replications.
- Provide objective compliance evidence against repo contracts.
- Provide a pre-registered replication spec (commands, thresholds, artifacts).

## Standards Anchors
- `README.md`
- `docs/standards/MEASUREMENT_CONTRACT.md`
- `docs/REPRODUCIBILITY_POLICY.md`
- `docs/CANONICAL_EXPERIMENTS.md`
- `docs/PIPELINE_OPERATIONS.md`

## Package Contents
- `INDUSTRY_GRADE_COMPLIANCE.md`: pass/fail checklist with evidence paths.
- `SEED_BRIDGE_REPLICATION_SPEC.md`: pre-registered protocol for seed bridge replication.
- `STATUS.json`: machine-readable gate status.
- `run_seed_bridge_matrix.sh`: execution script for the 15-run seed matrix.
- `run_seed_bridge_matrix_hardened.sh`: resume+retry+error-capture variant.
- `analyze_seed_bridge_matrix.py`: post-run analysis (per-seed + pooled random-effects).
- `analyze_semantic_behavior.py`: embedding-based semantic rescoring (`all-MiniLM-L6-v2`, cosine thresholding).
- `recompute_final_correlations.py`: consolidates final seed-bridge + semantic correlation outputs.
- `collect_seed_results_from_remote.sh`: pull remote seed run artifacts and run analysis.
- `run_seed_bridge_dense_pythia_pilot.sh`: non-GQA dense pilot batch launcher.
- `run_seed_456_subfolder.sh`: targeted runner for `configs/canonical/seed_bridge_2026_02_20/seed_456/`.
- `analyze_multi_token_confounds.py`: truncation/confound analysis for multi-token bridge runs.
- `RUNTIME_LOCK.md`: runtime lock and drift notes.
- `DENSE_MODEL_PILOT_SPEC.md`: architecture-dependence pilot protocol.
- `HARDENING_CHANGELOG.md`: concrete hardening changes implemented in this pass.
- `evidence/verify_research_ready.txt`: readiness validator output.
- `evidence/artifact_contract_audit.json`: machine-readable artifact audit.
- `evidence/artifact_contract_audit.md`: human-readable artifact audit.
- `evidence/seed_batch_launch_utc.txt`: remote launch metadata for the seed batch.
- `evidence/seed_bridge_analysis.json`: analysis output (auto-updated from local + synced remote runs).
- `evidence/seed_bridge_analysis.md`: analysis summary.
- `evidence/multi_token_confound_analysis.json`: confound-focused stats.
- `evidence/multi_token_confound_analysis.md`: confound-focused summary.
- `evidence/semantic_behavior_analysis.json`: semantic rescoring aggregates for seed-bridge + C2.
- `evidence/semantic_behavior_analysis.md`: semantic rescoring summary.
- `evidence/final_correlations.json`: consolidated final correlation pack (seed bridge + semantic + C2).
- `evidence/final_correlations.md`: concise final correlation summary.
- `evidence/semantic_bridge_scores_seed_bridge.csv`: per-sample semantic scores for seed-bridge runs.
- `evidence/semantic_bridge_scores_c2.csv`: per-sample semantic scores for C2 outputs.
- `evidence/seed_bridge_paired_dotplot.png`: cross-seed paired mean plot (auto-written when plotting deps available).
- `evidence/runtime_snapshot_remote_2026-02-20.txt`: remote runtime snapshot.
- `configs/canonical/seed_bridge_2026_02_20/`: 15 pre-generated config files + `RUN_MATRIX.csv`.
- `configs/canonical/seed_bridge_dense_pythia_2026_02_20/`: 9 pre-generated dense pilot configs + `RUN_MATRIX.csv`.

## Current Gate Status
- Repo readiness: `PASS` (`RESULT: RESEARCH READY`)
- Artifact contract (bridge + multi-token synchronized runs): `PASS`
- Spec readiness for seed bridge replication: `READY`
- Seed bridge standout gate: `PASS` (seeds `42`, `123`, `456` all pass directional + significance criteria)
- Semantic rescoring gate: `COMPLETE` (full seed-bridge + C2 rescored; see `evidence/semantic_behavior_analysis.json` for mixed/condition-specific outcomes)

## Notes
- `src/pipelines/canonical/multi_token_bridge.py` was hardened to emit canonical summary keys `rv_cohens_d` and `rv_p_value`.
- Synced multi-token summary in `results/remote_gpu_sync/2026-02-20/.../summary.json` was canonicalized with the same keys for standards-consistent auditing.
