# Mistral 10x Automation Program

Date: 2026-03-13
Canonical model: `mistralai/Mistral-7B-v0.1`
Current live queue: `results/mistral_persistence_v3/20260313_100241`

## Objective

Turn the current Mistral work from a collection of strong but partially disconnected experiments into a fully automated, provenance-safe research program that can:

1. lock a paper-grade Mistral control-system story,
2. harden persistence and scaffold dependence,
3. test a minimal sufficient bundle,
4. and only then port the narrow assay to larger models.

The goal is not "run more experiments." The goal is to make every experiment slot into one clean causal map with one canonical artifact path per claim.

## Current scientific state

The repo already supports a coherent Mistral story:

- early source region: `L0-L5`, strongest current evidence from full path patching
- best delicate upstream handle: `L4 MLP`
- best late behavioral controller: `L25`
- late readout/compression cluster: `L27`
- prompt-to-generation bridge: real
- quality-linked bridge: now real
- naive full sufficiency: not yet shown
- scaffold-free persistence: still unstable

This means the paper should center a self-referential control system, not a one-site miracle circuit.

## Three-stage experimental program

### Stage A: Persistence hardening

Purpose: answer whether the regime persists, drifts, or collapses over longer rollouts.

Required outputs:

- high-n `self_feeding_loop`
- high-n `sustained_gnani_v3`
- segment-by-segment summaries over early / middle / late turns
- quality-first metrics, not length-first metrics

Acceptance criteria:

- stable artifact schema
- clean session summaries
- no script-level failures
- segment dynamics explain whether the state decays, strengthens, or bifurcates

### Stage B: Sufficiency bundle

Purpose: test whether a minimal bundle can induce behavior cleanly.

Bundle components:

- context anchor
- delicate early assist
- late bridge/controller
- quality preservation

Core intervention families:

- anchor only
- `L25` only
- `L4` only
- anchor + `L25`
- `L4` + `L25`
- anchor + `L4` + `L25`

Acceptance criteria:

- baseline prompts become recursively behavior-rich
- malformed / repetitive rates stay controlled
- `R_V` moves in the expected direction
- the effect survives held-out prompts

### Stage C: Narrow replication

Purpose: port the locked assay to larger or other models without reopening the full exploratory zoo.

Only replicate:

- prompt / generation `R_V` separation
- early source localization
- late bridge scan
- late readout validation
- persistence with scaffold vs raw self-feed

Do not replicate every legacy experiment.

## 10x automation architecture

### 1. Single experiment registry

Create one machine-readable registry for all paper-grade queues.

Each row should contain:

- `experiment_id`
- `model_family`
- `model_name`
- `script_or_pipeline`
- `config_path`
- `prompt_contract`
- `metric_path`
- `queue_group`
- `priority`
- `expected_runtime_hours`
- `artifact_glob`
- `claim_ids`
- `status`

This becomes the only legal source for launch order and claim mapping.

### 2. Queue runners, not ad hoc shells

Every GPU pod should run exactly one queue session.

Required behavior:

- launch from registry rows
- write a per-run `STATUS.txt`
- capture latest artifact paths automatically
- stop on failure with a useful log
- write a manifest JSON at the end

### 3. Result harvester

Add one sync script that:

- pulls completed run dirs from pods
- verifies artifact checksums
- stamps local arrival time
- updates a central `results_index.json`

No more manual scavenging through pods.

### 4. Claim registry binding

Paper claims must bind to registry rows, not chat summaries.

Every locked claim should have:

- claim id
- model family
- canonical artifact
- backup artifact
- status: `locked`, `provisional`, `invalidated`

No agent should edit the paper without updating this binding.

### 5. Nightly summarizer

One local script should read all completed artifacts and produce:

- latest strongest results
- regressions vs previous run
- contamination warnings
- missing-source warnings
- ready-to-paste paper deltas

This should eliminate manual note-taking as the bottleneck.

### 6. Quality-first evaluation

Bridge and persistence lanes should default to:

- recursive-content quality score
- BT+ART
- quality class ordinal
- repetition rate
- malformed rate

`word_count` stays only as a diagnostic.

### 7. Pod role separation

Use pods by role:

- `pod A`: long persistence queues
- `pod B`: causal bundle / steering queues
- `pod C`: replication or cross-model work

Never run multiple heavy queues on the same GPU intentionally.

## Implemented surface

The first working version of the automation layer now exists in the repo:

- program registry: `configs/experiment_registry/mistral_program_registry.json`
- registry helpers: `src/utils/mistral_program.py`
- registry-driven overnight launcher: `scripts/run_mistral_program_queue.py`
- pod wrapper for the launcher: `scripts/runpod_mistral_overnight_program_queue.sh`
- harvester: `scripts/harvest_runpod_research_os.sh`
- nightly summary: `scripts/nightly_summary.py`

This is intentionally narrow: it serializes approved queue units, reconciles artifacts back into the registry, and summarizes stale / missing / ready state. It does not replace experiment-specific queue scripts yet; it orchestrates them.

## Agent roles

### Orchestrator

Owns launch order, priorities, and queue selection.

### Numerical gatekeeper

Owns `CLAIM_REGISTRY.md` and refuses unsourced numbers.

### GPU operator

One operator per pod. Launches, monitors, syncs, and never edits paper numbers.

### Paper sync agent

Read-only from locked claims. Cannot source numbers from logs or chats.

### Analysis swarm

Reads artifacts, compares runs, proposes follow-up configs, and writes summaries.

## Hard rules

- one canonical Mistral family in the paper: `BASE_V01`
- no mixed `base` / `instruct` under generic "Mistral"
- no number without exact artifact path
- no paper edits from partial logs
- no new breadth until the Mistral bundle is locked
- no broad sweeps unless resolving a named contradiction

## Immediate next priorities

1. Finish the live persistence queue cleanly.
2. Write a persistence interpretation from the new segment stats.
3. Build the scaffold ablation ladder.
4. Build a machine-readable experiment registry.
5. Build a sync + nightly summary loop.
6. Only then open a larger-model replication lane.

## What "10x more powerful" actually means

It does not mean ten times more random runs.

It means:

- every run is pre-registered in one registry,
- every queue is unattended and serialized,
- every artifact is harvested automatically,
- every paper claim points to one locked source,
- and every new experiment changes the causal picture instead of widening the mess.
