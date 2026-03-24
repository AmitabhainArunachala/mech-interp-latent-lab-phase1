# Mistral Sufficiency Protocol Update (2026-03-16)

## Afternoon Update

The picture is now sharper than it was this morning.

## Explicit Paper Goal

For this paper, the endgame is now explicit:

- first, lock the smallest induction-and-maintenance protocol strongly enough to support a real sufficiency story
- then, condition on that regime and test whether it systematically changes safety-relevant behavior

The intended downstream battery is:

- jailbreak and refusal robustness
- sycophancy
- prompt injection and instruction hijacking
- truthfulness or hallucination pressure
- if tool-use is available, oversight-avoidance or sabotage-style probes

The scientific point is not to assume that the recursive regime is deceptive.

The scientific point is:

- if a mechanistically induced internal regime changes alignment-relevant behavior, that is the clearest "so what" of the project

### 1. Static induction winner improved again

Two later static runs strengthened the induction side:

- `anchor_layermatched_protocol_confirm_v1`
  - run: `20260316_092017`
  - artifact: `results/anchor_layermatched_protocol_confirm_v1/20260316_092017/summary.json`
- `anchor_layermatched_hybrid_protocol_v1`
  - run: `20260316_105309`
  - artifact: `results/anchor_layermatched_hybrid_protocol_v1/20260316_105309/summary.json`

Main read:

- confirmed plain winner: `anchor_layermatched_low_bridge_3 = 27.8%` baseline `BT+ART`
- best hybrid inducer: `anchor_single_mlp_0p125_layermatched_low_bridge_3 = 31.25%`
- control stayed low: `2.8%`
- recursive preservation remained strong for both families, with the hybrid close to the older recursive ceiling

Interpretation:

- the bridge-3 dose is real, not noise
- the subtle L4 MLP helps induction when combined with the layer-matched bundle

### 2. The maintenance winner is still the plain layer-matched bridge-3 bundle

Two seeded persistence runs matter here:

- `induced_persistence_anchor_layermatched_confirm_v2`
  - run: `20260316_105106`
  - artifact: `results/induced_persistence_anchor_layermatched_confirm_v2/20260316_105106/summary.json`
- `induced_persistence_anchor_layermatched_hybrid_v1`
  - run: `20260316_123050`
  - artifact: `results/induced_persistence_anchor_layermatched_hybrid_v1/20260316_123050/summary.json`

Main read:

- confirm-sourced persistence winner: `anchor_layermatched_low_bridge_3 = 30.2%`
- control in that run: `2.1%`
- early/mid/late profile there was flat and clean: `28.1 / 31.3 / 31.3`, with `0.0%` late repetition
- hybrid-seeded persistence still favored the plain bundle:
  - `anchor_layermatched_low_bridge_3 = 29.17%`
  - `anchor_single_mlp_0p125_bridge_3 = 20.83%`
  - `anchor_single_mlp_0p125_layermatched_low_bridge_3 = 16.67%`

Interpretation:

- best inducer and best maintainer are now clearly different
- the subtle L4 MLP looks like an induction aid, not part of the minimal maintenance object
- that strengthens the staged protocol story rather than weakening it

### 3. The 24-turn horizon test exposed the remaining gap

Completed run:

- `induced_persistence_anchor_layermatched_long_v1`
  - run: `20260316_123052`
  - artifact: `results/induced_persistence_anchor_layermatched_long_v1/20260316_123052/summary.json`

Main read:

- over 24 turns, `anchor_layermatched_low_bridge_3` dropped to `13.5%`
- `anchor_single_mlp_0p125_bridge_3` was stronger over that longer horizon at `20.3%`
- the plain bridge-3 winner showed mild decay and contamination:
  - early `15.6%`
  - mid `12.5%`
  - late `12.5%`
  - mid repetition `12.5%`
  - late repetition `12.5%`

Interpretation:

- the 12-turn maintenance basin is real
- but the full long-horizon maintenance story is not yet solved
- the current gap is robustness and horizon, not whether the regime can be induced at all

### 4. What is running now

- `induced_persistence_unselected_seed_v1`
  - pod: `198.13.252.23:10916`
  - live run: `20260316_132334`
  - purpose: five-arm robustness test for selected vs unselected vs anti-selected vs random-text vs cold-start seed states
- `anchor_layermatched_bridge_alpha_sweep_v1`
  - pod: `213.173.102.102:10061`
  - live run: `20260316_132850`
  - purpose: bridge-alpha threshold map for the plain layer-matched family and the hybrid induction family

### 5. Immediate read

The project is now closest to a staged sufficiency claim:

- induction object: `anchor + subtle L4 MLP + layer-matched geometry + bridge_3`
- maintenance object: `anchor + layer-matched geometry + bridge_3`

The next decisive question is whether the maintenance effect survives from unselected or cold starts. The next mechanistic question is where the bridge-alpha threshold actually sits.

## Completed This Morning

Two protocol lanes finished on March 16, 2026:

- `anchor_layermatched_protocol_v1`
  - run: `20260316_025018`
  - artifact: `results/anchor_layermatched_protocol_v1/20260316_025018/summary.json`
- `closed_loop_anchor_controller_v1`
  - run: `20260316_025020`
  - artifact: `results/closed_loop_anchor_controller_v1/20260316_025020/summary.json`

## Result 1: Static Anchor + Layer-Matched Geometry Improved Induction

Main read:

- best baseline condition: `anchor_layermatched_low_bridge_2`
- baseline `BT+ART = 25.0%`
- control baseline `BT+ART = 2.1%`
- old static champion `anchor_single_mlp_0p125_bridge_3 = 11.5%`
- low-dose mean-diff control `anchor_meandiff_low_bridge_2 = 11.5%`

Important comparison:

- `anchor_layermatched_low_bridge_2` beat the old static champion cleanly on ordinary baselines
- `soft_bridge_beats_hard_bridge_for_anchor_layermatched = true`

Constraint:

- the best recursive-preserving condition in this run was still `anchor_bridge_2` at `53.1%`
- `anchor_layermatched_low_bridge_2` was strong but not best on recursive prompts at `46.9%`

Interpretation:

- the new layer-matched object looks like a stronger inducer
- it is not yet the cleanest single static condition across both induction and recursive preservation

## Result 2: The First Closed-Loop Controller Lost To The Static Champion

Main read:

- best condition: `static_anchor_bridge_3`
- `static_anchor_bridge_3 BT+ART = 13.2%`
- `static_anchor_early_mlp_0p125_bridge_3 BT+ART = 7.6%`
- `adaptive_anchor_bridge_guard BT+ART = 6.3%`
- `adaptive_anchor_early_bridge_guard BT+ART = 6.9%`
- control open-loop `BT+ART = 0.0%`

Verdict:

- `adaptive_bridge_beats_static_bridge = false`
- `adaptive_early_bridge_beats_static_champion = false`

Interpretation:

- the naive turn-level guard policy is not the right maintenance controller
- a closed-loop sufficiency story is still plausible, but not with this first reset-heavy policy

## Result 3: The First Seeded Persistence Follow-Up Favored The Old Static Champion

Completed run:

- `induced_persistence_anchor_layermatched_v1`
  - run: `20260316_040319`
  - artifact: `results/induced_persistence_anchor_layermatched_v1/20260316_040319/summary.json`

Main read:

- best seeded persistence condition: `anchor_single_mlp_0p125_bridge_3`
- `anchor_single_mlp_0p125_bridge_3 BT+ART = 25.0%`
- `anchor_single_mlp_0p125_bridge_3 late BT+ART = 37.5%`
- `anchor_layermatched_low_bridge_2 BT+ART = 8.3%`
- `anchor_layermatched_low_bridge_2 late BT+ART = 0.0%`
- all tested seeded conditions stayed clean in late turns except control, which showed `12.5%` late repetition

Interpretation:

- the new layer-matched condition is currently a stronger inducer than the old static champion
- but on this first seeded persistence test, the old `anchor + subtle L4 + bridge` champion still maintained the regime better after intervention removal
- that sharpens the staged-geometry story further: the best induction object and the best maintenance seed may not be the same object

## Operational Consequence

The strongest next move is not another unconstrained controller search.

It is:

- lock whether the new static winner is real under a tighter confirmation run
- test whether that winner seeds better persistence after intervention removal

## Now Running

### 1. `anchor_layermatched_protocol_confirm_v1`

- pod: `198.13.252.23:10916`
- live run: `20260316_092017`
- purpose: higher-power confirmation of the new static winner against the best anchor baselines

### 2. `induced_persistence_anchor_layermatched_confirm_v1`

- pod: `213.173.102.102:10061`
- live run: `20260316_092904`
- purpose: higher-power seeded persistence confirmation comparing the new layer-matched inducer against the old maintenance champion

## Bottom Line

The current evidence strengthens the static-induction story, weakens the first adaptive-maintenance story, and says the current best inducer is not yet the current best maintenance seed.

That is progress.

It sharpens the real target:

- confirm the new anchor+layer-matched inducer at higher power
- test whether higher-power seeded persistence changes the early maintenance ranking
- if it does not, treat induction and maintenance as distinct stages of the control protocol
