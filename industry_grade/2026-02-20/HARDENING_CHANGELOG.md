# Hardening Changelog (2026-02-20)

## Completed in this pass

1. Runner contract enforcement
- File: `src/pipelines/run.py`
- Added explicit `MULTI_TOKEN_REQUIRED_KEYS` enforcement for `multi_token_bridge`.
- Added strict-mode checks for `multi_token_bridge` required keys.

2. Immutable artifact manifest
- File: `src/pipelines/run.py`
- Added `manifest.json` generation per run with SHA256 hashes for run artifacts.
- Manifest includes git commit, model/model_revision, prompt bank version, artifact sizes + hashes.

3. Multi-token canonical key hardening
- File: `src/pipelines/canonical/multi_token_bridge.py`
- Added canonical top-level fields:
  - `rv_cohens_d`
  - `rv_p_value`

4. Industry-grade dated package
- Folder: `industry_grade/2026-02-20/`
- Added compliance docs, status file, evidence bundle, runtime lock notes.

5. Seed matrix operational hardening
- Added `run_seed_bridge_matrix_hardened.sh` with:
  - resume semantics (`SUCCESS` skip)
  - retry policy (`MAX_RETRIES`)
  - per-attempt logs
  - per-attempt error artifact capture

6. CI merge gates
- Added:
  - `scripts/ci_validate_canonical_configs.py`
  - `scripts/ci_validate_runner_contract.py`
  - `.github/workflows/quality-gates.yml`
- Purpose: fail CI on canonical config drift or runner contract drift.

7. Deeper-science scaffolding
- Added dense non-GQA pilot matrix:
  - `configs/canonical/seed_bridge_dense_pythia_2026_02_20/` (9 runs)
  - `industry_grade/2026-02-20/DENSE_MODEL_PILOT_SPEC.md`
  - `industry_grade/2026-02-20/run_seed_bridge_dense_pythia_pilot.sh`
- Added low-truncation multi-token probe config:
  - `configs/canonical/multi_token_bridge_mistral_low_trunc_probe.json`
- Added confound analysis tool:
  - `industry_grade/2026-02-20/analyze_multi_token_confounds.py`

8. Activation bridge runtime resilience
- File: `src/pipelines/canonical/rv_l27_activation_patching_bridge.py`
- Added per-pair `try/except` so a single bad pair is logged and skipped instead of killing the run.
- Added periodic checkpoint writes (`checkpoint_every`, default 1) to persist `per_sample.csv` during long runs.
- Added `pair_errors.csv` artifact with pair index + traceback snippets.
- Added explicit progress heartbeat lines for long remote runs.
- Added optional `generation_timeout_sec` guard for `model.generate(...)` to fail fast on pathological generations.

9. Seed analysis robustness
- File: `industry_grade/2026-02-20/analyze_seed_bridge_matrix.py`
- Analysis now scans both:
  - `results/phase1_mechanism/runs/`
  - `results/remote_gpu_sync/2026-02-20/phase1_mechanism/`
- Uses paired tests on matched `(rec_id, base_id)` rows when available.
- Emits a machine-readable standout signal gate in `evidence/seed_bridge_analysis.json`.

10. Seed matrix config safety defaults
- Folder: `configs/canonical/seed_bridge_2026_02_20/`
- Added explicit:
  - `"generation_timeout_sec": 120`
  - `"checkpoint_every": 1`
- Synced updated configs to remote so remaining queued runs inherit safeguards.
- Added targeted runner for user-requested seed subfolder:
  - `industry_grade/2026-02-20/run_seed_456_subfolder.sh`
  - supports resume + retries + timeout guard.

11. Semantic behavioral scorer (embedding-based)
- Added: `industry_grade/2026-02-20/analyze_semantic_behavior.py`
- Uses `sentence-transformers/all-MiniLM-L6-v2`.
- Scoring rule:
  - semantic score = max cosine similarity to 5 fixed `L5_refined` exemplars from prompt bank.
  - semantic recursive if score `> 0.4`.
- Re-scored:
  - all discoverable `c2_rv_measurement.csv` outputs
  - all discoverable seed-bridge `per_sample.csv` outputs
- Emitted artifacts:
  - `evidence/semantic_behavior_analysis.json`
  - `evidence/semantic_behavior_analysis.md`
  - `evidence/semantic_bridge_scores_seed_bridge.csv`
  - `evidence/semantic_bridge_scores_c2.csv`

12. Cross-seed analysis + publication plot plumbing
- Enhanced `industry_grade/2026-02-20/analyze_seed_bridge_matrix.py`:
  - pooled paired t-tests across all completed seed overlaps
  - explicit CI + effect size in JSON/MD output
  - paired-dot plot artifact:
    - `evidence/seed_bridge_paired_dotplot.png`

## Validation executed
- `python3 scripts/verify_research_ready.py` -> PASS
- `python3 scripts/ci_validate_canonical_configs.py` -> PASS (warnings only)
- `python3 scripts/ci_validate_runner_contract.py` -> PASS
- `python3 -m py_compile ...` for modified Python files -> PASS

## In-flight
- Priority seed control batch is running remotely from:
  - `industry_grade/2026-02-20/run_seed_bridge_priority_signal.sh`
- Seed 42 full triad is complete and passes the standout gate in the expected direction.
