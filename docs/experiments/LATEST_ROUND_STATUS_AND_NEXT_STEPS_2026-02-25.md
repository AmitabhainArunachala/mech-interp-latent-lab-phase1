# Latest Round Status and Next Steps (2026-02-25)

## Where This Round Lands Us

Current causal position:

- **Necessary**: confirmed for dual-layer geometry (L18 residual + L27 V-proj).
- **Not sufficient**: still not shown.
- Best framing: **necessity without sufficiency**.

## Evidence Snapshot

Source artifacts:

- `results/persistent_patching_v3/persistent_patching_v3_dual_20260225_002604.json`
- `results/persistent_patching_v2/persistent_patching_v2_20260224_141952.json`
- `results/phase1_mechanism/runs/20260225_015414_c2_rv_measurement/c2_rv_measurement.csv`

v3 dual-layer (n=10 sessions/condition):

- BREAK: `56.0% -> 3.7%` BT+ART
  - Session-level exact permutation p = `1.62e-05`
  - Cohen's d = `3.12`
- INDUCE: `3.7% -> 0.3%` BT+ART (wrong direction at turn level)
  - Session-level exact permutation p = `0.334`
  - No induction support at session level

v2 single-layer (n=5 sessions/condition):

- BREAK: `65.3% -> 59.3%`, session permutation p = `0.652` (NS)
- INDUCE: `0.7% -> 4.0%`, session permutation p = `0.447` (NS)

C2 (n=50 prompts/config, paired by prompt index):

- baseline vs kv_only: delta R_V = `0.109`, paired t p = `7.05e-09`
- baseline vs c2_full: delta R_V = `0.397`, paired t p = `1.99e-32`
- kv_only vs c2_full: delta R_V = `0.288`, paired t p = `1.41e-27`

Interpretation:

- Geometry can be strongly moved.
- Behavior follows in break direction but not induce direction.
- Prompt/context pathways remain required.

## Caveat Fixed For Next Runs

Observed in v3: baseline-side R_V missingness was high due short generations:

- baseline_clean: `35.3%` R_V missing
- baseline_dual_patched: `40.0%` R_V missing

To reduce this confound, `scripts/sufficiency_ladder.py` now includes:

- `--min-new-tokens` (default `24`) to suppress early EOS.
- `--seed` for reproducibility.
- Block-randomized condition execution per session index to reduce time-order drift.
- R_V missing-rate diagnostics already logged.

## Immediate Next Experiment (RunPod)

Primary objective: test sufficiency with baseline 2x2 (`KV x dual_patch`).

Run:

```bash
bash scripts/gpu_batch_sufficiency_2x2.sh
```

Equivalent direct command:

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

Pre-registered pass gate (already implemented in output JSON):

- target: `kv_plus_dual` vs `clean_baseline`
- requires all:
  - lift >= `0.15`
  - turn-level Fisher p < `0.01`
  - direction = `UP`
  - session permutation p < `0.05`

## Decision Tree After Next Run

If pre-registered gate **passes**:

- Claim practical sufficiency for this intervention bundle.
- Then ablate bundle components for minimal sufficient set (already partially encoded by 2x2).

If gate **fails**:

- Keep necessity claim as primary.
- Next candidate additions for sufficiency test:
  - early-layer residual transfer (L0-L3),
  - stronger semantic donor KV protocol,
  - attention-pattern transfer controls.

## Paper Positioning (Recommended)

Lead claim:

- "We identify a mechanistic substrate that is causally necessary for recursive behavior."

Secondary claim:

- "Geometry transfer alone is not sufficient; prompt-conditioned pathways contribute irreducibly."

