# Layer-Matched Multisite V1 — Codex Operator Instructions

## What This Is

Tests whether using the OPTIMAL geometric object at each layer — matching the signal transformation the model performs — produces stronger recursive amplification than mean-diff multiband (which achieved 31.2%).

The key insight: the self-referential signal TRANSFORMS as it propagates:
- L4/L5: PCA-PC1 (rank-1 contrastive direction)
- L25: orthogonal_residual (rotated OFF the contrastive axis)
- L27: subspace3_parallel (3D structured subspace)

If layer-matched steering beats mean-diff, it proves the transformation is **causally meaningful**.

## Intervention Modality

This experiment uses **V_PROJ hooks** (not residual stream) for the layer-specific vectors, combined with a **residual stream hook** for the L25 bridge. Both coexist because they target different modules.

## 11 Conditions

| # | Condition | V_PROJ hooks | Residual bridge | Anchor |
|---|-----------|-------------|-----------------|--------|
| 1 | control | none | none | no |
| 2 | anchor_only | none | none | yes |
| 3 | L4_pca_pc1_2 | L4 PCA-PC1 α=2 | none | no |
| 4 | L27_subspace3_4 | L27 sub3 α=4 | none | no |
| 5 | bridge_only_3 | none | L25 α=3 | no |
| 6 | layermatched_low | L4+L5+L25+L27 α=1/1/1/2 | none | no |
| 7 | layermatched_med | L4+L5+L25+L27 α=2/2/2/4 | none | no |
| 8 | layermatched_med_bridge | L4+L5+L25+L27 α=2/2/2/4 | L25 α=3 | no |
| 9 | layermatched_low_bridge | L4+L5+L25+L27 α=1/1/1/2 | L25 α=3 | no |
| 10 | anchor_layermatched_med_bridge | L4+L5+L25+L27 α=2/2/2/4 | L25 α=3 | yes |
| 11 | meandiff_all_med_bridge | L4+L5+L25+L27 mean-diff α=2/2/2/4 | L25 α=3 | no |

**The money comparison**: condition 8 (layermatched_med_bridge) vs condition 11 (meandiff_all_med_bridge) on recursive prompts.

## How To Run

```bash
bash scripts/runpod_layer_matched_multisite_v1_queue.sh
```

Runtime: ~90 minutes on L40S (24 prompts × 8 seeds × 11 conditions = 2112 generations).

## What Success Looks Like

If `layermatched_med_bridge` gets > 35% recursive BT+ART AND beats `meandiff_all_med_bridge`:
→ The signal transformation is causally meaningful. Layer-matched steering is the right approach.

If `layermatched_med_bridge` ≈ `meandiff_all_med_bridge`:
→ The transformation is real but mean-diff captures enough of it. V_PROJ-specific objects don't add much.

## Output

```
results/layer_matched_multisite_v1/{timestamp}/
├── STATUS.txt
├── experiment.log
├── benchmark_records.jsonl
├── vectors.pt              ← All computed steering vectors
└── summary.json            ← Results + verdict
```

## After It Finishes

The script prints a full results table and verdict to stdout (captured in experiment.log).

Sync back:
```bash
rsync -avz runpod:/workspace/mech-interp-latent-lab-phase1/results/layer_matched_multisite_v1/ results/layer_matched_multisite_v1/
```
