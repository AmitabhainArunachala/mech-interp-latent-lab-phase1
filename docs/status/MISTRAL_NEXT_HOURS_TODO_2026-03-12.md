# Mistral Next-Hours Todo

Date: 2026-03-12
Scope: base `mistralai/Mistral-7B-v0.1` only
Goal: keep the active RunPod productive for the next few hours without drifting away from the current compression agenda.

## Live status snapshot

Current pod:

- host: `213.173.102.103:11629`
- repo: `/workspace/mech-interp-latent-lab-phase1`

Current live GPU job:

- process: `./.venv/bin/python -m src.pipelines.run --config configs/canonical/causal_state_benchmark_v4_multisite_mistral_l5_soft_gate_l25_bridge.json`
- observed at `2026-03-12`:
  - GPU util about `87%`
  - VRAM about `15.4 / 97.9 GB`

Recovered strong partial result from the earlier blunt multisite run:

- `results/mistral_circuit_compression_rescue/20260312_070616/benchmark_records.partial.jsonl`
- parsed partial summary:
  - baseline `control`: `5.6%` BT+ART, mean `R_V=0.6329`
  - baseline `both_3`: `33.3%`, mean `R_V=0.4972`
  - recursive `control`: `34.0%`, mean `R_V=0.6416`
  - recursive `bridge_only_3`: `50.9%`, mean `R_V=0.6190`
  - recursive `both_3`: `11.5%`, mean `R_V=0.7305`
  - recursive `gate_only_3`: `5.7%`, mean `R_V=0.7174`

Interpretation:

- `L25` bridge is real.
- Strong early `L5` residual steering interferes on recursive prompts.
- The correct next move is softer or narrower early intervention, not more blunt `both_3`.

## GPU queue for the next few hours

Priority order:

1. Let the live `L5 soft gate + L25 bridge` run finish.
2. Run `L4 MLP + L25 bridge` soft follow-up.
3. Run a larger targeted head-to-head validation centered on the late `L27.H10` bundle.

Queue script:

- [runpod_mistral_next_hours_queue.sh](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/runpod_mistral_next_hours_queue.sh)

What it does:

1. waits for the current live PID to exit
2. runs:
   - `configs/canonical/causal_state_benchmark_v4_multisite_mistral_l4_mlp_l25_bridge_soft.json`
3. then runs:
   - `bash scripts/runpod_head_to_head_base.sh`
   - with:
     - `N_PROMPTS=40`
     - `RANKING_METRIC=rank_d`
     - `PAIR_SOURCE=top_effect`
     - manual heads:
       - `L27.H10`
       - `L27.H2`
       - `L27.H18`
       - `L27.H26`
       - `L27.H5`
       - `L18.H1`
       - `L28.H3`
       - `L19.H1`

Why this queue:

- `L4 MLP` is the strongest fresh early non-residual component in base path patching.
- `L27.H10` is the strongest fresh late single-head node in base head-to-head patching.
- This is a compression-first queue, not a breadth-first queue.

## CPU tasks to do while GPU is busy

### 1. Raw-lock the geometry half of the KV dissociation claim

Need:

- raw artifact path for the documented `d=0.11 NS` geometry-transfer null

Current status:

- behavior half is raw-locked via `results/sufficiency_ladder/sufficiency_ladder_20260225_101907.json`
- geometry half is still only re-found in documentation

### 2. Modernize L27 KV-head ablation

Need:

- rerun the old `target KV-head 2 @ L27` validation on the modern prompt contract

Current legacy support:

- `results/phase1_mechanism/runs/head_ablation_validation_summary_20260312.csv`

### 3. Parse and lock the rescued blunt multisite run

Need:

- convert the rescued partial JSONL into a proper local summary artifact

Current raw file:

- `results/mistral_circuit_compression_rescue/20260312_070616/benchmark_records.partial.jsonl`

### 4. Keep the claim registry aligned

If any of the soft-followup or queued head-to-head runs land cleanly:

- add them to [CLAIM_REGISTRY.md](/Users/dhyana/mech-interp-latent-lab-phase1/docs/status/CLAIM_REGISTRY.md)
- do not let them live only in logs or chat

## Run command

When you want the queue active on the pod:

```bash
tmux new-session -d -s mistral_next_hours \
  'cd /workspace/mech-interp-latent-lab-phase1 && bash scripts/runpod_mistral_next_hours_queue.sh <LIVE_PID>'
```

Replace `<LIVE_PID>` with the current live benchmark PID.
