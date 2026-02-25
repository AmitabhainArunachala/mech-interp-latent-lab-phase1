# Pipeline Operations (GPU / Canonical Runner)

This is the operational runbook for running experiments **reproducibly** in this repo.

## Single blessed entrypoint

All reproducible runs go through the config-driven runner:

```bash
python -m src.pipelines.run --config <path/to/config.json>
```

## Recommended first runs (canonical)

These are the fastest paper-grade checks to run on a fresh GPU session:

```bash
# Layer-27 causal necessity for R_V (geometry-only; fastest)
python -m src.pipelines.run --config configs/canonical/rv_l27_causal_validation.json --strict

# Activation patching bridge (geometry-only)
python -m src.pipelines.run --config configs/canonical/rv_l27_activation_patching_bridge.json

# KV cache patching bridge (often slower than activation patching)
python -m src.pipelines.run --config configs/canonical/rv_l27_kv_patching_bridge.json
```

## Multi-token bridge (slower)

If you are explicitly testing the **behavior ↔ geometry** connection:

```bash
python -m src.pipelines.run --config configs/canonical/multi_token_bridge_mistral.json
```

## Where outputs go

Runs write to a timestamped directory:

```
results/<phase>/runs/<YYYYMMDD_HHMMSS>_<experiment>[_<run_name>]/
```

Each run directory contains (minimum):
- `config.json`: atomic snapshot of the config used
- `summary.json`: machine-readable summary metrics (schema enforced for canonical experiments)
- `report.md`: human-readable report that embeds the summary JSON
- `prompt_bank_version.json`: prompt bank hash (from `prompts/bank.json`)
- `metadata.json`: standardized run metadata (git commit, params, measurement contract info)
- `hardware_info.json`: best-effort hardware report
- `error.txt`: only if the run failed

Additionally, the runner appends a global ledger:
- `results/RUN_INDEX.jsonl`

## Strict mode (recommended for canonical)

For canonical experiments, use `--strict` when you want to enforce “no missing required metrics”.

```bash
python -m src.pipelines.run --config configs/canonical/confound_validation.json --strict
```

Notes:
- Geometry-only canonical experiments intentionally do not generate text, so `logit_diff_*` may be `null`.
- Strict mode excludes geometry-only experiments from “logit_diff must be present”.

## Choosing / authoring configs

- Canonical configs: `configs/canonical/`
- Archived/historical configs: `configs/archive/`

Config schema (top-level):

```json
{
  "experiment": "rv_l27_causal_validation",
  "params": { "model": "...", "seed": 42, "n_pairs": 80 },
  "results": { "root": "results", "phase": "phase1_mechanism" }
}
```

The `experiment` key must exist in the registry:
- `src/pipelines/registry.py`

## Contract + standards

Before changing measurement logic, read:
- `docs/standards/MEASUREMENT_CONTRACT.md` (LOCKED)
- `docs/CANONICAL_EXPERIMENTS.md` (required metrics + geometry-only vs behavioral split)

