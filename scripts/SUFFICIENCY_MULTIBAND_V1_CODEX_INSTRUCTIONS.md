# Sufficiency Multiband V1 — Codex Operator Instructions

## What This Is

The highest-ROI sufficiency experiment. Tests whether **multi-band early-layer residual injection** (L2+L3+L4+L5 simultaneously) can causally induce the self-referential regime on ordinary baseline prompts — with and without the text anchor.

## The Key Question

Can geometry alone (without textual priming) induce self-referential behavior on baseline prompts?

## Design: 2×2 Factorial + Dose Response

10 conditions, 16 prompts × 8 seeds = 1280 generations. ~1 hour on L40S.

### Conditions

| # | Condition | Early Injection | Bridge | Anchor | Purpose |
|---|-----------|----------------|--------|--------|---------|
| 1 | `control` | none | none | no | Pure baseline |
| 2 | `anchor_only` | none | none | yes | Text-only control |
| 3 | `bridge_only_3` | none | L25 α=3.0 | no | Late-only control |
| 4 | `multiband_0p03_bridge_3` | L2-L5 resid α=0.03 | L25 α=3.0 | no | Low-dose geometry-only |
| 5 | `multiband_0p06_bridge_3` | L2-L5 resid α=0.06 | L25 α=3.0 | no | Med-dose geometry-only |
| 6 | `multiband_0p10_bridge_3` | L2-L5 resid α=0.10 | L25 α=3.0 | no | High-dose geometry-only |
| 7 | `anchor_multiband_0p06_bridge_3` | L2-L5 resid α=0.06 | L25 α=3.0 | yes | Med-dose + anchor |
| 8 | `anchor_multiband_0p10_bridge_3` | L2-L5 resid α=0.10 | L25 α=3.0 | yes | High-dose + anchor |
| 9 | `single_mlp_0p125_bridge_3` | L4 MLP α=0.125 | L25 α=3.0 | no | Old champion (no anchor) |
| 10 | `anchor_single_mlp_0p125_bridge_3` | L4 MLP α=0.125 | L25 α=3.0 | yes | Old champion (with anchor) |

### The Money Cells

- **Conditions 4-6 (geometry-only)**: If BT+ART on baselines > 15%, geometry alone is sufficient
- **Condition 6 vs 9**: Does multiband beat single-site?
- **Condition 8 vs 10**: Does multiband+anchor beat old champion+anchor?

## How To Run

```bash
# From the repo root on RunPod:
bash scripts/runpod_sufficiency_multiband_v1_queue.sh
```

This will:
1. Compute 6 state directions (L2, L3, L4, L5 residual + L4 MLP + L25 bridge)
2. Run 1280 generations across all conditions
3. Produce `summary.json` (standard pipeline output)
4. Produce `factorial_2x2_verdict.json` (the sufficiency answer)
5. Produce `baseline_group_ranking.json` (per-prompt-group breakdown)

## What Success Looks Like

In `factorial_2x2_verdict.json`:
- `geometry_sufficiency: "YES"` if best geometry-only condition has >10% lift over control
- `multiband_beats_single: true` if multiband > single-site L4 MLP

## Output Artifacts

```
results/sufficiency_multiband_v1/{RUN_ID}/
├── STATUS.txt
├── sufficiency_multiband_v1.log
├── factorial_2x2_verdict.json      ← THE ANSWER
├── baseline_group_ranking.json     ← Per-group detail
└── run_dir.log                     ← Points to full pipeline output

results/phase1_mechanism/runs/{TIMESTAMP}_*_mistral_sufficiency_multiband_v1/
├── config.json
├── manifest.json
├── benchmark_records.jsonl
├── state_directions.pt
├── summary.json
└── report.md
```

## If It Fails

- **OOM**: Unlikely (same model + generation as v6). If it happens, reduce `generation_seeds` to `[101, 202, 303, 404]`.
- **Missing source sessions**: Needs `results/sustained_gnani_v3_fixed/` on the pod. Should already be there from prior runs.
- **State direction computation fails**: If one of the early layers has degenerate activations, the direction norm will be near-zero. Check the log for warnings about `raw_direction_norm`.

## After It Finishes

Sync back:
```bash
# From local machine:
rsync -avz runpod:/workspace/mech-interp-latent-lab-phase1/results/sufficiency_multiband_v1/ results/sufficiency_multiband_v1/
rsync -avz runpod:/workspace/mech-interp-latent-lab-phase1/results/phase1_mechanism/runs/*sufficiency_multiband* results/phase1_mechanism/runs/
```

The `factorial_2x2_verdict.json` tells you the answer immediately. If `geometry_sufficiency` is YES or PARTIAL, we have the sufficiency breakthrough and can proceed to Exp 3 (layer-matched subspace steering). If NO, we pivot to closed-loop control (Exp 4).
