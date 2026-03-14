# Mistral Sufficiency Bundle Program (2026-03-13)

## Purpose

This program turns the current base-Mistral story into a mostly unattended experimental sequence.

The central hypothesis is no longer "R_V alone is sufficient." The current repo evidence supports a
smaller and sharper claim:

- `L0-L5` is an early source region
- `L4 MLP` is the best delicate upstream handle found so far
- `L25` is the strongest late behavior-control handle
- `L27` is a late readout/compression cluster
- context anchoring matters, so strict sufficiency must be tested as a bundle, not as a single-site edit

## Current State

### Locked

- `results/path_patching/path_patching_summary_20260312_125939.json`
  - strongest early sites: `L5 residual d=4.152`, `L5 v_proj d=2.540`, `L4 mlp d=2.123`
  - strongest late specific site: `L27 v_proj d=-1.941`
- `results/phase1_mechanism/runs/20260312_133759_head_ablation_validation_mistral_l27_kv2_modern_core_measurement__summary.json`
  - modern `L27` validation on frozen contract: `d=4.51`, `p=4.19e-49`
- `results/phase1_mechanism/runs/20260312_133909_rv_l27_kv_patching_bridge__summary.json`
  - geometry moves toward recursive without significant behavior rescue
- `results/phase1_mechanism/runs/20260312_141514_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_confirmation_window8/summary.json`
  - best confirmed behavior lift: `47.2%` recursive vs `44.4%` bridge-only
- `results/phase1_mechanism/runs/20260312_150039_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_confirmation_window4/summary.json`
  - best confirmed lower-leak tradeoff: `44.4%` recursive with `8.3%` baseline spillover

### Discovery-Only

- `results/phase1_mechanism/runs/20260312_133738_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_micro4_l25_bridge/micro_window_summary.json`
  - strongest discovery condition: `early_mlp_0p03125_bridge_3`
  - recursive `52.8%`, baseline `2.8%`, recursive mean output `R_V=0.6336`

This `micro4` result is interesting but not yet paper-grade on its own. It should be treated as a promotion candidate until reconfirmed under an independent follow-up.

## Program Structure

### Stage A: Strict Bundle Confirmation

Goal: prove the best subtle `L4 MLP + L25` combinations are real and reproducible.

Status:

- mostly complete
- `window8` and `window4` are the current confirmation runs

Decision rule:

- use the best condition that improves recursive `BT+ART` beyond `bridge_only_3`
- prefer lower baseline spillover on ties
- if the discovery-only `micro4` winner matters for the paper, rerun it independently before promoting it

### Stage B: Persistence Hardening

Goal: determine whether the bundle meaningfully affects generation beyond prompt-time and short behavioral steering.

This stage has three linked sub-questions:

1. Does prompt-time geometry still predict generation-time behavior when truncation is reduced?
2. Does the recursive regime remain cleaner over long generation horizons?
3. Does contextual scaffold matter more than direct self-feeding?

This is where the program hardens:

- multi-token bridge
- low-truncation bridge
- long-generation bridge
- self-feeding loop
- sustained gnani

### Stage C: Integrated Sufficiency Decision

The bundle is strong enough for the paper only if all of the following are true:

- short-horizon induction is better than `L25` alone or materially safer at similar lift
- generation-time bridge evidence survives lower truncation
- scaffolded long-turn behavior clearly outperforms raw self-feeding
- geometry and behavior are described honestly when they diverge

If Stage C fails, the paper should still claim:

- necessity
- partial controllability
- delicate upstream assist
- context dependence

It should not claim full sufficiency.

## Automation Design

### Remote Queue

Use:

- `scripts/runpod_mistral_sufficiency_bundle_queue.sh`

This runs sequentially on the pod and writes:

- `results/mistral_sufficiency_bundle/<run_id>/STATUS.txt`
- `results/mistral_sufficiency_bundle/<run_id>/bundle_manifest.json`

Default stages:

- bridge hardening
- self-feeding loop
- sustained gnani

Optional:

- strict L4 confirmation reruns via `STRICT_STAGE=1`

### Local Launcher

Use:

- `scripts/launch_runpod_mistral_sufficiency_bundle.sh`

Required environment:

- `RUNPOD_HOST`
- `RUNPOD_PORT`

Optional:

- `RUNPOD_SSH_KEY`
- `REMOTE_REPO`
- `REMOTE_SESSION`
- `SYNC_CODE_FIRST=0` if the remote repo is already current

Example:

```bash
RUNPOD_HOST=root@198.13.252.23 \
RUNPOD_PORT=16717 \
bash scripts/launch_runpod_mistral_sufficiency_bundle.sh
```

This:

1. rsyncs the local code to the pod
2. starts a dedicated tmux session
3. launches the remote queue
4. prints monitor commands

## Bridge Hardening Configs

New configs for this program:

- `configs/canonical/multi_token_bridge_mistral_confirmatory_n20.json`
- `configs/canonical/multi_token_bridge_mistral_low_trunc_probe_n12.json`
- `configs/canonical/multi_token_bridge_mistral_longgen_n12.json`

These extend the earlier bridge lane instead of inventing a new contract.

## Hardening Fix Applied

`src/pipelines/canonical/multi_token_bridge.py` now records both:

- all-valid H1 correlation
- non-truncated H1 correlation

This keeps the bridge summary from quietly depending on a selection-biased headline.

## Operational Rules

- one GPU, one heavy queue
- do not launch multiple long jobs manually on the same pod
- discovery results do not become paper claims without an independent confirmation run
- every promoted claim must cite one raw artifact path

## Recommended Next Pod Session

If a new pod is available now, the best unattended run is:

1. `bridge_confirmatory_n20`
2. `bridge_low_trunc_n12`
3. `bridge_longgen_n12`
4. `self_feeding_loop`
5. `sustained_gnani_v3`

This is the shortest path to clarifying whether the `L4/L25/L27` bundle is only a short-horizon controller or the front end of a genuinely persistent self-referential regime.
