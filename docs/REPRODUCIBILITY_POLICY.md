# Reproducibility Policy

This repo treats reproducibility as a **contract**: same code + same config + same prompt bank → same results (within expected floating-point tolerance).

## What counts as a reproducible run

A run is considered reproducible if it produces a complete artifact bundle with enough metadata to be re-run and audited.

### Required invariants

- **Config-driven execution**: run via `python -m src.pipelines.run --config ...`
- **Prompt bank versioning**: record prompt bank hash from `prompts/bank.json`
- **Seed control**: fixed `seed` in config for deterministic runs
- **Evaluation mode**: `model.eval()` and `torch.no_grad()` for measurement
- **Measurement contract compliance**: R_V measured on **prompt tokens** (not generated text) unless explicitly a “dynamic/temporal” experiment

### Required artifacts

The run directory must include:
- `config.json` (atomic snapshot)
- `summary.json` (machine-readable metrics + stats)
- `report.md` (human-readable wrapper)
- `prompt_bank_version.json`
- `metadata.json` (standardized run metadata)
- `hardware_info.json` (best-effort)

The global ledger is append-only:
- `results/RUN_INDEX.jsonl`

## Dependency policy

This repo uses two dependency files:
- `requirements.txt`: development-friendly (minor version movement allowed)
- `requirements.lock`: pinned direct dependencies for “bit-perfect” reproduction (as close as practical)

GPU environments typically require a CUDA-specific PyTorch wheel; see `README.md` for the recommended install line.

## Canonical vs archive

- **Canonical experiments** are intended to meet publication-grade expectations and to be validated by `--strict` summary-schema checks in `src/pipelines/run.py`.
- **Archive experiments** are historical. They may still run, but are not guaranteed to satisfy current summary-schema requirements.

## Measurement contract (authoritative)

The locked measurement definition lives here:
- `docs/standards/MEASUREMENT_CONTRACT.md`

If you need to change definitions, you must:
- bump the contract version
- document the change
- ensure configs + summaries record the new version

## Statistical reporting expectations

For any between-group claim (recursive vs baseline), summary outputs should include:
- `n_pairs`
- group means (recursive, baseline)
- delta
- Cohen’s d
- p-value

Canonical experiments enforce required summary keys; see:
- `docs/CANONICAL_EXPERIMENTS.md`
- `src/pipelines/run.py` (required keys list)

