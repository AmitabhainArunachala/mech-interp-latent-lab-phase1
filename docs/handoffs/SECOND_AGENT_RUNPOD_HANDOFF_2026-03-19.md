# Second Agent RunPod Handoff

Date: 2026-03-19
Purpose: use a second hot GPU without colliding with the main Mistral sufficiency branch

## Current Mainline State

Do not interfere with the existing primary Mistral run.

Primary pod branch already in flight:

- experiment: `positive_broad_persistence_confirm_v1`
- queue group: `mistral_positive_broad_persistence_confirm_v1`
- live pod: `grotesque_beige_salmon`
- live tmux: `mistral_positive_broad_persistence_confirm`

That branch is the highest-ROI Mistral causal story right now.

The second agent should therefore use the second pod for one of:

1. cross-architecture locked replication
2. paper-hardening replications on other base models

Not for a duplicate Mistral persistence run.

## Recommended Priority

Use the second pod in this order:

1. `llama3_8b_p0_canonical_v1`
2. `gemma9b_p0_canonical_v1`
3. `mixtral8x7b_p0_canonical_v1` only if the pod is clearly large enough and stable

Why this order:

- `Llama-3-8B` is the safest low-friction cross-architecture replication.
- `Gemma-2-9B` is still high-value and should fit on a strong A100 pod.
- `Mixtral-8x7B` is the most likely to hit memory / download / runtime limits, so it should not be the first thing the second agent tries.

## Scientific Goal

The second pod is not chasing a new Mistral story.

Its job is to strengthen the paper around:

- cross-architecture heterogeneity with locked provenance
- the claim that the main Mistral effect is real, but not trivially universal
- clean upgraded artifacts for the appendix / robustness section

## Logging Contract

Follow this exactly:

- [AMIROS_RUNPOD_LOGGING_STANDARD_2026-03-18.md](/Users/dhyana/mech-interp-latent-lab-phase1/docs/handoffs/AMIROS_RUNPOD_LOGGING_STANDARD_2026-03-18.md)

Minimum acceptable deliverables per run:

- `STATUS.txt`
- step log files
- `summary.json` or equivalent canonical artifact
- `research_os lease-update` state
- `research_os result-upsert` state

## Pod Environment

Before launching any queue wrapper, set:

```bash
export AMIROS_POD_NAME="<human-readable-pod-name>"
export AMIROS_HOST="<public-ip-or-ssh-host>"
export AMIROS_PORT="<ssh-port>"
export AMIROS_SESSION="<tmux-session-name>"
```

Example:

```bash
export AMIROS_POD_NAME="new_crossmodel_pod"
export AMIROS_HOST="1.2.3.4"
export AMIROS_PORT="12345"
export AMIROS_SESSION="llama3_8b_p0_canonical_v1"
```

## Launch Order

### Lane 1: Llama-3-8B

Use:

- [runpod_llama3_8b_p0_canonical_v1_queue.sh](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/runpod_llama3_8b_p0_canonical_v1_queue.sh)

Launch in tmux:

```bash
tmux new-session -d -s llama3_8b_p0_canonical_v1 \
  'cd /workspace/mech-interp-latent-lab-phase1 && bash scripts/runpod_llama3_8b_p0_canonical_v1_queue.sh'
```

This run already performs:

- canonical P0 measurement
- full path patching
- AMIROS lease updates
- AMIROS result upserts

### Lane 2: Gemma-2-9B

Use:

- [runpod_gemma9b_p0_canonical_v1_queue.sh](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/runpod_gemma9b_p0_canonical_v1_queue.sh)

Launch only after Llama is clearly healthy or done:

```bash
tmux new-session -d -s gemma9b_p0_canonical_v1 \
  'cd /workspace/mech-interp-latent-lab-phase1 && bash scripts/runpod_gemma9b_p0_canonical_v1_queue.sh'
```

### Lane 3: Mixtral-8x7B

Use only if:

- the pod has enough VRAM
- downloads are stable
- the agent is not already fighting rate limits or OOM

Use:

- [runpod_mixtral8x7b_p0_canonical_v1_queue.sh](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/runpod_mixtral8x7b_p0_canonical_v1_queue.sh)

Launch:

```bash
tmux new-session -d -s mixtral8x7b_p0_canonical_v1 \
  'cd /workspace/mech-interp-latent-lab-phase1 && bash scripts/runpod_mixtral8x7b_p0_canonical_v1_queue.sh'
```

## Failure Policy

If a model hits a limit:

- keep `STATUS.txt`
- keep the failing step log
- let the wrapper record `FAIL`
- do not hand-edit AMIROS JSON files
- do not silently retry forever

Then move down the priority list:

- if `Mixtral` fails, drop to `Gemma` or `Llama`
- if `Gemma` fails, drop to `Llama`

The second agent should prefer completed clean artifacts over heroic unstable runs.

## Sync-Back Procedure

After each run:

1. `scp` the remote artifact directory back into local `results/`
2. regenerate the dashboard:

```bash
cd /Users/dhyana/mech-interp-latent-lab-phase1
python3 scripts/nightly_summary.py
```

3. verify the canonical board:

- [AMIROS_STATUS_BOARD.md](/Users/dhyana/mech-interp-latent-lab-phase1/docs/status/AMIROS_STATUS_BOARD.md)

## What To Report Back

The second agent should return:

- model
- run id
- whether P0 completed
- whether path patching completed
- artifact paths
- one-paragraph scientific read
- any blocker such as auth, OOM, or malformed outputs

## Copy-Paste Prompt For The Second Agent

```text
You are using a second RunPod for mech-interp experiments in /workspace/mech-interp-latent-lab-phase1.

Your job is not to branch the main Mistral sufficiency story. The primary Mistral persistence confirm is already running elsewhere. Your job is to strengthen cross-architecture evidence with clean AMIROS logging.

Follow exactly:
- docs/handoffs/AMIROS_RUNPOD_LOGGING_STANDARD_2026-03-18.md
- docs/handoffs/SECOND_AGENT_RUNPOD_HANDOFF_2026-03-19.md

Priority order:
1. scripts/runpod_llama3_8b_p0_canonical_v1_queue.sh
2. scripts/runpod_gemma9b_p0_canonical_v1_queue.sh
3. scripts/runpod_mixtral8x7b_p0_canonical_v1_queue.sh only if the pod is clearly large enough and stable

Requirements:
- run each experiment in its own tmux session
- preserve STATUS.txt, step logs, result-upsert, and lease-update state
- sync finished artifacts back to local results/
- regenerate docs/status/AMIROS_STATUS_BOARD.md after sync

Do not invent new logging formats. Do not manually edit registry JSON unless absolutely necessary. Prefer completed clean provenance over risky retries.
```

## Bottom Line

The main pod is currently chasing the strongest Mistral slam-dunk.

The second pod should add clean cross-model evidence with strict logging, starting with `Llama-3-8B`, then `Gemma-2-9B`, and only then `Mixtral-8x7B` if resources allow.
