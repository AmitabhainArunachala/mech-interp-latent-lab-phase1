## Executive summary (today)

We established **model-specific grounding** for `mistralai/Ministral-8B-Instruct-2410` (36 layers) and then mapped a **behavioral collapse “button”** under residual injection.

Key discovery: **the behavioral-collapse layers (L24–L27) are not the same as the strongest \(R_V\) separation corridor (≈L30–L35) in this model**.

This implies at least two distinct coordinates:
- **Geometric corridor** (where baseline vs recursive \(R_V\) separates the most)
- **Behavioral sensitivity** (where residual patches readily induce collapse/attractor-like repetition)

---

## What we ran

### 1) Phase 1 existence sweep (Ministral-native)

- **Model**: `mistralai/Ministral-8B-Instruct-2410` (36 layers)
- **Run**: `results/phase1_mechanism/runs/20251213_120917_phase1_existence_ministral8b_instruct_phase1_existence_fast_v1/`
- **Artifact**: `phase1_layer_sweep.csv`
- **Finding**: strongest separation layers are late (e.g. L30/L33/L35), especially for window=32.

### 2) Collapse-layer map (behavior)

We tested whether baseline generation collapses after a push-only residual patch.

- **Model**: `mistralai/Ministral-8B-Instruct-2410`
- **Settings**: sampling on, `temperature=0.7`, **window=32**
- **Layers tested**: {24, 26, 27, 30, 33, 35}
- **Finding**: **L24 and L27 consistently cause collapse**, L26 is less stable, and L30/33/35 largely do not.

### 3) n≈65 per layer batch (stopped early; enough signal)

- **Run**: `results/phase1_mechanism/runs/20251213_124735_behavioral_grounding_batch_ministral8b_n100_L24_27_W32_sampled_v1/`
- **Planned**: 100 pairs × 4 layers × 2 conditions = 800 rows
- **Actual**: 520 jsonl rows → **65 paired baselines** per layer (24–27)

#### Aggregate results (paired baseline vs baseline_patched; window=32)

| layer | n_pairs | mean repeat4 (base) | mean repeat4 (patched) | mean Δrepeat4 | frac(patched repeat4>0.75) | mean Δunique |
|------:|--------:|--------------------:|------------------------:|--------------:|---------------------------:|-------------:|
| 24 | 65 | 0.0957 | 0.5516 | +0.4559 | 0.5385 | -0.2452 |
| 25 | 65 | 0.0582 | 0.4452 | +0.3870 | 0.4308 | -0.1764 |
| 26 | 65 | 0.0773 | 0.4833 | +0.4060 | 0.4615 | -0.1814 |
| 27 | 65 | 0.0892 | 0.4660 | +0.3768 | 0.4462 | -0.1559 |

Interpretation: **patching at L24–L27 produces large, reliable increases in repetition and decreases in lexical diversity**, with L24 strongest on average.

---

## What this most likely means (careful language)

- There exists a **behavioral attractor / failure mode** that can be triggered by residual injection at L24–L27 on Ministral-8B-Instruct.
- This **does not automatically mean** “the recursion mechanism lives at L24–L27.”
  - It may instead indicate a **vulnerable control interface** (a “phase knob”) that, when perturbed, destabilizes decoding into repetitive loops (“What?” / “The The The …”).
- Separately, \(R_V\) separation is strongest later (≈L30–L35), suggesting **a geometric corridor** that is not identical to the behavioral collapse interface.

---

## Most hopeful finding (why this is exciting)

On a stronger 36-layer model, we found a **clean dissociation**:

- **Geometry peak layer** (R_V separation): ~L30–L35
- **Behavioral collapse layer** (patch-induced degeneration): ~L24–L27

This is *good news* because it suggests we’re not just measuring “degeneracy” with \(R_V\). Instead, we likely have:
- a real geometric order parameter corridor, and
- an earlier control surface where residual perturbations can flip the system into an attractor.

That is a tractable mechanistic story.

---

## Suggested next steps (high priority)

1. **Add “RV-on-generated” via teacher-forcing** on a subset:
   - take prompt + generated continuation, run a forward pass, compute PR/RV on the generated suffix window.
2. **Separate “collapse” vs “recursion” behaviors**:
   - add a “degeneracy score” (repeat loops) and a “self-ref score” that excludes words like “process”.
3. **Compare controls for the patch source**:
   - use random/shuffled/opposite patches as sources at L24–L27 to confirm causality robustness (in this behavioral setting).
4. **Test whether patch strength matters**:
   - scale patch magnitude α (0→1) and see if collapse has a threshold (“barrier height”).
5. **Map the layer interaction**:
   - do a small two-patch experiment: push at L24 vs undo at L30–L35 (does the geometric corridor rescue collapse?).

---

## Things to set aside (low priority for now)

1. **Mixtral replication on this specific RunPod**:
   - blocked by disk quota/caching; do later on a better storage setup.
2. **Full “minimal circuit” IOI-style search right now**:
   - premature until we disentangle geometry vs collapse.
3. **Head-level attribution on Ministral before controls**:
   - likely noisy; better after we lock the behavioral metric + RV-on-generated pipeline.
4. **More prompt crafting (“new champions”)**:
   - we already have enough triggers; current bottleneck is mechanistic interpretation/controls.
5. **Expanding to many new model families immediately**:
   - first stabilize the cross-model protocol (find RV corridor + find collapse interface + run controls).











