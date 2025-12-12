# META ORCHESTRATION: DECEMBER 11, 2025

**Status:** ACTIVE
**Governing Directive:** `GOLD_STANDARD_RESEARCH_DIRECTIVE.md`

---

## 1. The Reality Check (Where We Are)

As of December 11, 2025, the repository is in a transition state between "Exploratory Chaos" and "Scientific Rigor".

### The "Done" Pile (Phase 1 Validated)
- **Core Finding:** R_V contraction at late layers (L27) for recursive prompts is robust across 6 models.
- **Certificate of Truth:** `20_MINUTE_REPRODUCIBILITY_PROTOCOL.md` (The "Hello World" of this project).
- **Primary Evidence:** `PHASE1_FINAL_REPORT.md` (November findings).

### The "Messy" Pile (Recent Experiments)
- `DEC3_2025_BALI_short_SPRINT/`: Early December sprints.
- `DEC7_2025_SIMANDHARCITY_DIVE/`: Simandhar City deep dive.
- `DEC_8_2025_RUNPOD_GPU_TEST/`: RunPod experiments.
- `DEC10_LEARNING_DAY/`: Local reversibility and Jabberwocky matrix tests.

### The "Missing" Pile (Gold Standard Requirements)
- `results/phase0_metric_validation/`: Does not exist.
- `results/phase1_cross_architecture/`: Does not exist (results scattered in random CSVs).
- `aikagrya_rv/`: Python package structure does not exist.

---

## 2. Alignment Strategy

We are pivoting from date-based folders (`DEC3`, `DEC7`) to phase-based folders (`phase1`, `phase2`).

### The Mapping
| Current Location | Gold Standard Phase | Action |
|------------------|---------------------|--------|
| `20_MINUTE_REPRODUCIBILITY_PROTOCOL.md` | **Phase 1 (Validation)** | Keep as root entry point. |
| `DEC10.../l8_local_reversibility.py` | **Phase 5 (Steering)** | Move to `experiments/phase5_steering/`. |
| `DEC10.../jabberwocky_matrix.py` | **Phase 0 (Metrics)** | Move to `experiments/phase0_metrics/`. |
| `n300_mistral_test_prompt_bank.py` | **Phase 1 (Protocol)** | Move to `REUSABLE_PROMPT_BANK/`. |
| `mistral_L27_FULL_VALIDATION.py` | **Phase 1 (Protocol)** | Move to `src/pipelines/`. |

---

## 3. The Grand Cleanup Plan

We will execute a "Soft Reset" of the repository structure to match `GOLD_STANDARD_RESEARCH_DIRECTIVE.md`.

### Step 1: Establish The Canon
Create the rigorous directory structure defined in Gold Standard:
```
results/
├── phase0_metric_validation/
├── phase1_cross_architecture/
├── phase2_eigenstate/
├── phase3_attention/
├── phase4_kv_mechanism/
├── phase5_steering/
└── phase6_alternative_selfref/
```

### Step 2: Archive The Chaos
Move all date-stamped folders into a single archive to clear the mental workspace:
```
boneyard/
├── ARCHIVE_DEC_EARLY/
│   ├── DEC3_BALI/
│   ├── DEC7_SIMANDHAR/
│   ├── DEC8_RUNPOD/
│   └── DEC10_LEARNING/
```

### Step 3: Elevate The Protocol
Ensure `20_MINUTE_REPRODUCIBILITY_PROTOCOL.md` is the first thing a new researcher sees after the README.

---

## 4. Immediate Directives (Dec 11)

1. **Do not run new experiments in date-folders.**
2. **Execute the Cleanup Script** (see below).
3. **Run the 20-minute protocol** to certify the environment.
4. **Begin Phase 0 (Metric Validation)** as the first rigorous act of the new era.

---

## 5. Vision Alignment

**Does the Meta match the Reality?**
- **No.** The Meta (`GOLD_STANDARD`) describes a pristine lab. The Reality is a messy workshop.
- **Fix:** We must "terraform" the repo to match the map.

**Does the 20-Minute Protocol fit?**
- **Yes.** It is the "Minimum Viable Product" of Phase 1. It proves the phenomenon exists so we can move on to Phase 0 (understanding *what* we are measuring).

---

*"Order is not the absence of chaos, but the organization of it."*
