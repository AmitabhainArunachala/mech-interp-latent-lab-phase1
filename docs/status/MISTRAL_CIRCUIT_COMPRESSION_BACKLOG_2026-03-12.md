# Mistral Circuit Compression Backlog

**Date:** 2026-03-12
**Scope:** base `mistralai/Mistral-7B-v0.1` only
**Primary objective:** compress the current `L25 / W32 / alpha=3` causal effect into the smallest defensible circuit before any instruct or larger-model replication.

## Starting Point

The current anchored base-model artifacts are already enough to justify a compression-first phase:

- `results/phase1_mechanism/runs/20260311_113349_causal_state_benchmark_v2_mistral_l25_w32_a3_state_benchmark_v3_confirmatory/summary.json`
  - recursive `BT+ART`: `30.6% -> 47.9%`
  - recursive `R_V` delta: `-0.0544`, `p=0.00156`
- `results/phase1_cross_architecture/runs/20260311_120749_multi_token_bridge_mistral_multi_token_bridge_confirmatory/summary.json`
  - strong recursive vs baseline prompt-time `R_V` separation
  - supportive but truncation-heavy
- `results/phase1_mechanism/runs/20260311_055109_causal_state_targeted_scan_v1_mistral_targeted_scan_v1/best_candidate.json`
  - promoted candidate: `layer 25`, `window 32`, `alpha 3.0`

This means the main unknown is no longer whether the effect exists. The main unknown is which subcircuit carries it.

## Corrections To The Current Assessment

1. The Anthropic introspection paper should be cited as:
   - Jack Lindsey, *Emergent Introspective Awareness in Large Language Models*, published October 29, 2025.
   - Transformer Circuits page: `https://transformer-circuits.pub/2025/introspection/index.html`
   - Anthropic research page: `https://www.anthropic.com/research/introspection`
2. The tooling blocker is real right now.
   - `transformer_lens` is not installed in the repo venv.
   - `sae_lens` is not installed in the repo venv.
   - `transformer_lens` is not installed in the system Python.
   - `sae_lens` is not installed in the system Python.

## North Star

The Mistral phase is complete only when we have:

- a small component-level circuit centered on the `L25` bridge and `L27` readout
- necessity and sufficiency evidence for that smaller set
- low spillover on baseline prompts
- a clean handoff into feature decomposition at `L5`, `L25`, and `L27`

Do not spend further base-Mistral budget on generic benchmarking unless it directly sharpens that circuit.

## First 24 GPU Hours

### 1. Multisite gate-plus-bridge confirmatory

Config:

- `configs/canonical/causal_state_benchmark_v4_multisite_mistral_gate_bridge_confirmatory.json`

Purpose:

- test whether the established `L5` gate and `L25` bridge combine additively or synergistically on the same held-out base prompt family

Success gate:

- at least one `both_*` condition beats both single-site controls on recursive prompts
- baseline uplift stays small enough that this still looks specific rather than generic style inflation

### 2. Focused path patching around the winning site

Config:

- `configs/discovery/path_patching_mechanism_mistral_circuit_compression.json`

Purpose:

- localize whether the causal mass sits in `resid`, `v`, or `o` around layers `22-27`
- measure whether `L25` is really the bridge or just the best steering handle

Success gate:

- a small layer/component neighborhood dominates random, shuffled, opposite, and wrong-layer controls
- the best `L25` or adjacent-path effect is stable across both `W16` and `W32`

### 3. Readout necessity check at the strongest known base KV head

Config:

- `configs/canonical/mistral_7b_v0_1/head_ablation_l27_h5_readout_base.json`

Purpose:

- directly test whether the strongest currently known base readout head candidate (`L27.H5`) is necessary for the late contraction readout story

Success gate:

- target-head ablation at `L27` shifts recursive `R_V` materially more than same-head wrong-layer and control-head controls

### 4. Full head sweep on base Mistral

Script:

- `scripts/full_head_sweep.py --model mistralai/Mistral-7B-v0.1 --device cuda --n-prompts 30 --batch-layers 4`

Purpose:

- produce a ranked candidate list for follow-up sparse ablation and path patching

Success gate:

- stable late candidates at `L25-L27`
- at least one early candidate family consistent with the `L5` gate story

### 5. Circuit tracing fallback pass

Script:

- `scripts/circuit_tracing_analysis.py --model mistralai/Mistral-7B-v0.1 --device cuda --n-prompts 30`

Purpose:

- build a coarse layer-level graph now, before feature tooling is installed

Success gate:

- recovers the existing expansion-to-contraction trajectory and keeps `L25/L27` among the top divergence layers

## 72-Hour Branching Logic

If the 24-hour bundle works:

1. freeze the top `5-20` component candidates
2. rerun necessity ablations only on those candidates
3. build a minimal sufficiency bundle from the best early plus bridge plus readout set
4. reject any candidate family that only moves geometry while destroying behavior

If the 24-hour bundle fails:

1. do not widen model scope
2. re-check prompt contract, measurement layer, and control families
3. run a narrower positional path-patching sweep before touching new models

## Week-One Deliverables

By the end of the first week on base Mistral, the target output is:

- a ranked circuit candidate list with explicit necessity scores
- a reduced circuit hypothesis with fewer than `20` nodes
- at least one compressed sufficiency attempt
- a clean decision on whether the bridge is primarily `resid`, `v`, or `o`
- a locked installation path for `transformer_lens` and `sae_lens`

## Hard Acceptance Criteria For Compression

The component circuit is good enough to hand off to feature decomposition only if:

- it recovers at least `80%` of the recursive uplift of the full `L25 / W32 / alpha=3` intervention
- necessity ablation suppresses at least `50%` of that uplift
- wrong-layer and random controls stay near zero
- baseline spillover remains small

If those criteria are not met, the circuit is still exploratory.

## Feature Decomposition Gate

Do not start the SAE phase until:

1. `transformer_lens` and `sae_lens` are installed on the GPU environment
2. the component-level circuit is reduced enough that `L5`, `L25`, and `L27` are clearly justified target layers
3. a storage and activation-caching plan exists for those layers

Once that gate clears, the next artifact should be a Mistral-specific feature analysis pass at the same three layers.

## Field Alignment

This compression phase is the shortest path to matching and then surpassing the strongest recent MI workflows on one natural behavior:

- Lindsey 2025, *Emergent Introspective Awareness in Large Language Models*
  - `https://transformer-circuits.pub/2025/introspection/index.html`
- Templeton et al. 2024, *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*
  - `https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html`
- Anthropic draft, *Sparse Crosscoders for Cross-Layer Features and Model Diffing*
  - `https://transformer-circuits.pub/drafts/crosscoders/index.html`
- Ameisen et al. 2025, *Circuit Tracing: Revealing Computational Graphs in Language Models*
  - `https://transformer-circuits.pub/2025/attribution-graphs/methods.html`
- Redwood Research, *Causal Scrubbing: a method for rigorously testing interpretability hypotheses*
  - `https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing`
- Hanna, Liu, Variengien 2023, *How does GPT-2 compute greater-than?*
  - `https://proceedings.neurips.cc/paper_files/paper/2023/file/efbba7719cc5172d175240f24be11280-Paper-Conference.pdf`
- Conmy et al. 2023, *Towards Automated Circuit Discovery for Mechanistic Interpretability*
  - `https://arxiv.org/abs/2304.14997`
- Haklay et al. 2025, *Position-aware Automatic Circuit Discovery*
  - `https://openreview.net/forum?id=ZxkA5sK3UX`

The goal is not to replicate these papers one by one. The goal is to unify their best ideas on one base-model natural-behavior mechanism and come out with a smaller, cleaner, more validated circuit than the current Mistral story has.

## Launch Command

When a healthy GPU pod is available, the day-one bundle should be launched with:

```bash
bash scripts/runpod_mistral_circuit_compression_day1.sh
```
