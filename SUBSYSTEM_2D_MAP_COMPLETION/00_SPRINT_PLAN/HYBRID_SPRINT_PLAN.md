# HYBRID SPRINT PLAN - 2D MAP COMPLETION + PYTHIA PILOT

## CONTEXT

We discovered two fundamental computational regimes (Retrieval vs. Computation) in validation testing. Before running full Pythia sweep (154 checkpoints × 6 models), we need to:

1. Complete the 2D subsystem map (add creative/planning/uncertainty)

2. Run 3-checkpoint Pythia pilot to validate emergence hypothesis

## TIMELINE: 6 HOURS ACTIVE COMPUTE

### PHASE 1: COMPLETE 2D MAP (3 hours)

**Test remaining subsystems with R_V + Attention Entropy**

| Subsystem | Prediction | Prompts | Time |
|-----------|-----------|---------|------|
| Creative | (0.6, 0.75) - High entropy | 5 constraint-based | 1h |
| Planning | (0.6, 0.45) - Moderate entropy | 5 sequential | 1h |
| Uncertainty | (1.0, 0.70) - Neutral + high entropy | 5 probabilistic | 1h |

**Outcomes:**

- **If separate:** 2D sufficient, launch full Pythia
- **If collapsed:** Need 3D metrics before Pythia
- **If uncertainty new regime:** Partial Pythia + 3D work

### PHASE 2: PYTHIA PILOT (2 hours)

**Test emergence on 3 checkpoints only**

| Checkpoint | Model | Purpose | Expected |
|------------|-------|---------|----------|
| 0 | pythia-2.8b | Untrained baseline | Random, R_V ≈ 1.0 |
| 76 | pythia-2.8b | Mid-training | Structure emerging? |
| 154 | pythia-2.8b | Final | Clean (0.6, 0.23) |

**Prompt:** Meta-cognitive (known working signature)

### PHASE 3: DECISION (Evening)

**Based on Phase 1 + 2 results:**

- Path A: Full Pythia sweep (if 2D clean)
- Path B: 3D metrics first (if collapsed)
- Path C: Hybrid (if partial separation)

## HARDWARE

RTX 6000 Blackwell - 102GB VRAM

- Can run both phases in parallel if needed
- No scarcity constraints

## SUCCESS CRITERIA

By end of day:

✓ Complete 2D map (7 subsystems)
✓ Pythia emergence tested (proof of concept)
✓ Clear decision for next week

---

*Jai Sat Chit Anand*

