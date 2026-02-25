# Sufficiency 2x2 Runbook (2026-02-25)

## Goal
Test whether combining KV swap with dual-layer geometry transfer is sufficient to induce recursive behavior on baseline prompts.

## Factorial Design
Baseline prompts only:

- `clean_baseline`: KV off, dual patch off
- `kv_only`: KV on, dual patch off
- `dual_patch`: KV off, dual patch on
- `kv_plus_dual`: KV on, dual patch on

Control:

- `clean_recursive`

## Script
- Runner: `scripts/sufficiency_ladder.py`
- Batch wrapper: `scripts/gpu_batch_sufficiency_2x2.sh`

## Default Command
```bash
python3 scripts/sufficiency_ladder.py \
  --n-sessions 10 \
  --max-turns 30 \
  --seed 42 \
  --rv-window 16 \
  --min-new-tokens 24 \
  --induce-min-lift 0.15 \
  --induce-alpha 0.01
```

## Pre-Registered Gate
Target comparison: `kv_plus_dual` vs `clean_baseline`

Pass requires all:

1. Turn-level BT+ART lift >= `0.15`
2. Turn-level Fisher p < `0.01`
3. Direction = `UP`
4. Session-level permutation p < `0.05`

The script writes this decision in:
- `comparisons.preregistered_decision`

## Output
- Result JSON: `results/sufficiency_ladder/sufficiency_ladder_<timestamp>.json`
- Batch log: `results/sufficiency_ladder/batch_sufficiency_<timestamp>.log`

## Notes
- Both turn-level and session-level stats are saved.
- RV null-rate (`n_rv_missing`) is tracked per condition/session.
- Condition execution is block-randomized per session index to reduce time-order confounds.
- `--min-new-tokens` suppresses early EOS so RV measurement has enough tokens and NaN rate stays low.
