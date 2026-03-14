# Mistral Numeric Source Of Truth

**Date:** 2026-03-11

> Superseded in part by
> `docs/handoffs/CODEX_BASE_V01_CRITICAL_BRIEFING.md` on 2026-03-12.
> In particular: do not treat the Instruct-v0.2 P0/head-sweep/path-patching lineage
> as canonical support for a paper that claims base `mistralai/Mistral-7B-v0.1`.

## Purpose

Prevent silent mixing of `mistralai/Mistral-7B-v0.1` and `mistralai/Mistral-7B-Instruct-v0.2`
results in paper drafting, sprint notes, or figure generation.

## Hard Rules

1. Never swap numbers between base and instruct Mistral runs.
2. Every claimed number must cite:
   - exact model variant
   - exact run artifact path
   - exact metric family
3. If two runs disagree, newer does **not** automatically win.
   The winner is the run from the intended benchmark family.
4. Until explicitly unified, `base v0.1` and `Instruct v0.2` support different claims.

## Benchmark Families

### Family A: Base Mistral (`mistralai/Mistral-7B-v0.1`)

Use for:
- historical causal-state benchmark lineage
- gate discovery (`v2`)
- multisite gate+bridge (`v4`)
- any benchmark whose config is authored for base Mistral

Current active run:
- pod 1
- tmux: `gate_bridge_scan`
- run id: `20260311_121159`

Canonical config paths:
- `configs/canonical/causal_state_benchmark_v2_gate_L3.json`
- `configs/canonical/causal_state_benchmark_v2_gate_L5.json`
- `configs/canonical/causal_state_benchmark_v2_gate_L7.json`
- `configs/canonical/causal_state_benchmark_v4_multisite_L5_L25.json`

### Family B: Instruct Mistral (`mistralai/Mistral-7B-Instruct-v0.2`)

Use for:
- hardened prompt-contract reruns
- `full_head_sweep`
- `svd_circuit_decomposition`
- `full_path_patching`
- hardened dual-patch diagnostics

Current active run:
- pod 2
- tmux: `mistral_n100`
- run id: `20260311_115454`

Canonical runner:
- `scripts/runpod_mistral_control_system_n100.sh`

## Current Known Mismatch Classes

### Dual-layer causal numbers

These are not interchangeable:

- `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_160920.json`
  - model: `mistralai/Mistral-7B-Instruct-v0.2`
  - old aggregated block
  - no degeneration fields
  - recursive clean `BT+ART = 0.4067`

- `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_191654.json`
  - model: `mistralai/Mistral-7B-Instruct-v0.2`
  - newer chat-template path
  - recursive clean `BT+ART = 0.5000`

- `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_204100.json`
  - model: `mistralai/Mistral-7B-Instruct-v0.2`
  - newest local synced instruct artifact
  - recursive clean `BT+ART = 0.5467`
  - recursive dual patched `BT+ART = 0.0000`
  - includes `mean_alpha_ratio`, `malformed_rate`, `repetitive_rate`

Interpretation:
- these three files are all **Instruct** lineage
- none of them should be relabeled as base-Mistral causal benchmark numbers
- old paper numbers like `40.7% -> 0.0%` and newer RunPod numbers like `54.7% -> 0.0%`
  are different instruct-family runs, not a single stable benchmark

### Head sweep counts

Counts such as `606/1024`, `630/1024`, or `691/1024` may differ because of:
- different model variant
- different counting criterion
- different metric or threshold family
- different run lineage

Do not cite any head-count headline without naming the exact artifact.

## Paper-Use Guidance

### Safe now

- signed-direction claims when exact artifact is cited
- instruct-family hardening claims for prompt-pass, head sweep, SVD, and full path patching
- base-family gate discovery claims once current pod-1 runs finish

### Not safe yet

- one single “definitive” dual-layer number across all Mistral variants
- mixing gate-discovery base numbers with instruct hardening numbers in one table
- treating instruct hardening results as replacements for base benchmark-family claims

## Operational Rule For The Current Sprint

1. Pod 2 results update the **Instruct hardening** story.
2. Pod 1 results update the **Base gate/bridge** story.
3. As of 2026-03-12, any paper section that explicitly claims base `mistralai/Mistral-7B-v0.1`
   must use base-family artifacts or be marked unsafe until rerun.

## Immediate Drafting Rule

If a section in `v006` uses a Mistral number, annotate it internally as one of:
- `BASE_BENCHMARK`
- `INSTRUCT_HARDENED`
- `MIXED_DO_NOT_USE`

Any number in the third category must be removed or replaced before paper freeze.
