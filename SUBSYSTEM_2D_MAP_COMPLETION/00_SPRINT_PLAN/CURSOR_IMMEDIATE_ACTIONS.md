# CURSOR: IMMEDIATE SETUP ACTIONS

## 1. CREATE FOLDER STRUCTURE

```bash
cd ~/Desktop
mkdir -p SUBSYSTEM_2D_MAP_COMPLETION/{00_SPRINT_PLAN,01_PROMPT_BANKS,02_CODE,03_RESULTS/{phase1_creative,phase1_planning,phase1_uncertainty,phase2_pythia_pilot},04_FIGURES}
cd SUBSYSTEM_2D_MAP_COMPLETION
```

## 2. COPY WORKING CODE FROM PREVIOUS SESSION

```bash
# Copy the validated pipeline from Jupyter/RunPod

# We need:
# - compute_participation_ratio()
# - compute_attention_entropy()
# - analyze_prompt_enhanced()
# - Model loading code
```

## 3. CREATE PROMPT BANK FILES

See: `01_PROMPT_BANKS/` - JSON files with 5 prompts each

## 4. SET UP MEASUREMENT PIPELINE

Create: `02_CODE/enhanced_pipeline.py`

- Loads pythia-2.8b
- Runs batch of prompts
- Outputs: R_V, Attention Entropy per prompt
- Saves: JSON results + CSV summary

## 5. CREATE VISUALIZATION SCRIPT

Create: `02_CODE/visualization.py`

- Reads all result JSONs
- Generates 2D scatter plot (R_V × Entropy)
- Colors by subsystem
- Saves to `04_FIGURES/`

## 6. PYTHIA PILOT SCRIPT

Create: `02_CODE/pythia_pilot.py`

- Loads checkpoints 0, 76, 154
- Tests single meta-cognitive prompt
- Tracks R_V emergence
- Quick validation (2 hours max)

## 7. READY STATE

When complete:

- Folder structure exists
- Code is executable
- Prompts are loaded
- Ready for John to run on RunPod

---

**Cursor, execute steps 1-7. Prepare everything. Report when ready.**

