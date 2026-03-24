# AMIROS RunPod Logging Standard

Use this exact contract when another agent launches experiments on a new pod.

## Goal

Every remote run should leave behind:

- a stable `run_id`
- a machine-readable `summary.json`
- raw per-example records when applicable
- a human-readable `STATUS.txt`
- step logs
- AMIROS lease state
- AMIROS result-index state

That is what keeps the local repo, remote pod, and paper claims synchronized.

## Required Environment

Set these before launching a queue script on the pod:

```bash
export AMIROS_POD_NAME="<human-readable pod name>"
export AMIROS_HOST="<public ip or ssh host>"
export AMIROS_PORT="<ssh port>"
export AMIROS_SESSION="<tmux session name>"
```

Example:

```bash
export AMIROS_POD_NAME="grotesque_beige_salmon"
export AMIROS_HOST="103.196.86.173"
export AMIROS_PORT="13805"
export AMIROS_SESSION="mistral_soft_break_chain"
```

## Directory Contract

Each queue wrapper should create both:

```text
results/<queue_group>/<run_id>/
results/<output_stem>/<run_id>/
```

Expected contents:

- `results/<queue_group>/<run_id>/STATUS.txt`
- `results/<queue_group>/<run_id>/<step>.log`
- `results/<output_stem>/<run_id>/summary.json`
- `results/<output_stem>/<run_id>/benchmark_records.jsonl` when the experiment emits row-level records

If `queue_group == output_stem`, one directory is fine in practice, but keep the same file contract.

## STATUS.txt Contract

At minimum, write:

```text
run_id=<RUN_ID>
python_bin=<PYTHON_BIN>
started_utc=<ISO8601 UTC timestamp>
>>> START <step_name> <ISO8601 UTC timestamp>
>>> DONE  <step_name> <ISO8601 UTC timestamp>
finished_utc=<ISO8601 UTC timestamp>
```

If a step fails:

```text
>>> FAIL <step_name> rc=<code> <ISO8601 UTC timestamp>
```

## AMIROS Lease Updates

At queue boot:

```bash
python -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group "$QUEUE_GROUP" \
  --run-id "$RUN_ID" \
  --status running \
  --current-step queue_boot \
  --out-dir "${OUT_DIR#$REPO_ROOT/}" \
  --notes "$NOTES"
```

Before each major step:

```bash
python -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group "$QUEUE_GROUP" \
  --run-id "$RUN_ID" \
  --status running \
  --current-step "$STEP_NAME" \
  --out-dir "${OUT_DIR#$REPO_ROOT/}"
```

On success:

```bash
python -m src.utils.research_os lease-update \
  --pod-name "$AMIROS_POD_NAME" \
  --host "$AMIROS_HOST" \
  --port "$AMIROS_PORT" \
  --session-name "$AMIROS_SESSION" \
  --queue-group "$QUEUE_GROUP" \
  --run-id "$RUN_ID" \
  --status completed \
  --current-step finished \
  --out-dir "${OUT_DIR#$REPO_ROOT/}"
```

On failure, use `--status failed`.

## AMIROS Result Upsert

At the end of a successful run:

```bash
python -m src.utils.research_os result-upsert \
  --run-id "$RUN_ID" \
  --queue-group "$QUEUE_GROUP" \
  --experiment-id "$EXPERIMENT_ID" \
  --status completed \
  --artifact-path "${RUN_OUT#$REPO_ROOT/}/summary.json" \
  --model-family BASE_V01 \
  --model-name mistralai/Mistral-7B-v0.1 \
  --config-path "<script + key args>" \
  --prompt-contract "<prompt family name>" \
  --metric-path "<what was measured>" \
  --claim-id "<claim id>"
```

## Summary Contract

Every `summary.json` should include:

- `timestamp`
- `experiment`
- `model`
- key config fields
- per-condition metrics
- a top-level `verdict`

For benchmark-style runs, keep:

- `conditions`
- `verdict`
- control metrics
- winner / best condition
- any selectivity or drop metrics used to choose the winner

## Raw Record Contract

If the run is prompt- or session-based, also write `benchmark_records.jsonl` with one JSON row per generation or turn.

Recommended fields:

- prompt id / prompt group / prompt mode
- generation seed
- condition
- generated text
- generated token count
- classification
- core metrics such as `bt_art`, `malformed`, `repetitive`, `output_rv`

## tmux Convention

Launch each long remote run in its own tmux session.

Good examples:

- `mistral_soft_break_sweep`
- `mistral_soft_break_tokenwindow`
- `mistral_soft_break_factorized`

Avoid reusing generic names like `run` or `exp`.

## Local Sync

After the remote run completes:

1. `scp` the artifact directory back into the local repo.
2. Run:

```bash
python3 scripts/nightly_summary.py
```

This regenerates [AMIROS_STATUS_BOARD.md](/Users/dhyana/mech-interp-latent-lab-phase1/docs/status/AMIROS_STATUS_BOARD.md).

## Best Template Source

If the agent wants a working queue-wrapper example, copy one of these patterns:

- [runpod_mistral_soft_break_latebundle_sweep_v1_queue.sh](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/runpod_mistral_soft_break_latebundle_sweep_v1_queue.sh)
- [runpod_mistral_soft_break_tokenwindow_v1_queue.sh](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/runpod_mistral_soft_break_tokenwindow_v1_queue.sh)
- [runpod_mistral_soft_break_factorized_v1_queue.sh](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/runpod_mistral_soft_break_factorized_v1_queue.sh)

## Bottom Line

Do not send back only prose summaries.

The minimum acceptable deliverable from a remote pod run is:

- `STATUS.txt`
- `summary.json`
- row-level records if applicable
- lease updates
- result upsert

If those exist, the run is usable.
