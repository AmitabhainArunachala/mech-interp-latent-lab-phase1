# RUN INDEX (Signal + Compliance)

## Purpose
Single source of truth for run provenance, artifact completeness, and audit status.

## Legend
- **Status**: KEEP / RAMP_UP / ARCHIVE
- **Issues**: Missing artifacts, contract violations, confounds

| Run / File | Experiment | Status | Key Stats | Artifacts | Issues / Notes |
|---|---|---|---|---|---|
| `results/phase1_cross_architecture/runs/20260202_121604_rv_l27_causal_validation_mistral_7b/summary.json` | rv_l27_causal_validation | KEEP | d=-2.259, p=2.24e-19, n=45 | config+summary+pairs+report+prompt_bank_version | Missing hardware_info.json |
| `results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/summary.json` | rv_l27_causal_validation | KEEP | d=-1.836, p=3.73e-16, n=45 | config+summary+pairs+report+prompt_bank_version | Missing hardware_info.json |
| `results/phase1_cross_architecture/runs/20260202_130807_rv_l27_causal_validation_gpt2_xl/summary.json` | rv_l27_causal_validation | KEEP | d=-1.143, p=6.15e-10, n=45 | config+summary+pairs+report+prompt_bank_version | Missing hardware_info.json |
| `results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/summary.json` | rv_l27_causal_validation | RAMP_UP | d=-0.719, p=8.7e-6, n=45 | config+summary+pairs+report+prompt_bank_version | R_V > 1.0; verify early/late layers + hardware_info |
| `results/phase1_cross_architecture/runs/20260202_115958_rv_l27_causal_validation_pythia_1_4b/summary.json` | rv_l27_causal_validation | RAMP_UP | d=-0.311, p=0.021, n=45 | config+summary+pairs+report+prompt_bank_version | p not <0.01; rerun n>=80 |
| `results/canonical/rv_l27_causal_validation/20251216_061127_rv_l27_causal_validation/summary.json` | rv_l27_causal_validation | RAMP_UP | p=2.75e-22, n=45 | summary+pairs | Missing d/CI + hardware_info |
| `results/canonical/confound_validation/20251216_060911_confound_validation/summary.json` | confound_validation | RAMP_UP | p=2.16e-06, n=37 | summary+csv | n<50 + missing d/CI + hardware_info |
| `results/canonical/multi_token_bridge/summary.json` | multi_token_bridge | RAMP_UP | H3 r=-0.308, p=6.19e-4 | summary+report+verdict | 89–90% truncation; canonical points to non-canonical run |
| `results/phase3_bridge/gemma_2_9b/.../summary.json` | multi_token_bridge | RAMP_UP | d=3.37, p=1.10e-35 | summary+report | Truncation high; missing hardware_info |
| `results/canonical/c2_measurement_suite/20260111_130123_c2_rv_measurement/summary.json` | c2_rv_measurement | RAMP_UP | rv_mean diff | summary+csv | Missing stats + hardware_info |
| `results/canonical/final_results.json` | rv_l27_causal_validation (legacy) | ARCHIVE | Values ~5–7 | summary only | Contract violation: single-layer PR mislabeled as R_V |

## Required Next Updates
- Populate missing `hardware_info.json` on all KEEP/RAMP_UP runs.
- Resolve Qwen2 layer selection and R_V definition audit.
- Re-run multi-token bridge with low truncation, log per-token R_V.
