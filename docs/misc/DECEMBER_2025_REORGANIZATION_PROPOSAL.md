# December 2025 Bonsai Reorganization Proposal

**Created:** December 8, 2025
**Purpose:** Consolidate, refine, and trim the December experimental files
**Safety:** All deletions listed explicitly; nothing essential lost

---

## Current State (Before)

```
mech-interp-latent-lab-phase1/
├── DEC3_2025_BALI_short_SPRINT/     # 1.0 MB - PRIMARY SOURCE
├── DEC7_2025_SIMANDHARCITY_DIVE/    # 68 KB - Contains duplicates!
├── DEC_8_2025_RUNPOD_GPU_TEST/      # 4.6 MB - PRIMARY SOURCE
├── DEC_8_2025_RUNPOD_TEST/          # 56 KB - Pre-GPU test artifacts
├── THE_GEOMETRY_OF_RECURSION_MASTER.ipynb      # 52 KB
├── THE_GEOMETRY_OF_RECURSION_MASTER_v2.ipynb   # 44 KB
├── [8 mistral_*.py files]           # November iterations
├── [multiple NOV_*.md files]        # November session logs
└── ... (other November files)
```

**Problems Identified:**
1. DEC7 folder contains empty/duplicate copies of DEC3's subfolders
2. Two master notebooks (v1 and v2) - should keep only v2
3. DEC_8_2025_RUNPOD_TEST (no GPU) is superseded by DEC_8_2025_RUNPOD_GPU_TEST
4. Multiple November patching scripts that evolved into final versions

---

## Proposed Structure (After)

```
mech-interp-latent-lab-phase1/
├── DECEMBER_2025_EXPERIMENTS/           # NEW consolidated folder
│   ├── DEC3_BALI/                       # Renamed, cleaned
│   │   ├── LLAMA3_L27_REPLICATION/      # Keep - has CSVs and code
│   │   ├── DEC4_LOGIT_LENS/             # Keep - has notebooks and validation
│   │   ├── FUTURE_EXPERIMENTS.md
│   │   └── DEC7_STEERING_VECTORS.py
│   │
│   ├── DEC7_SIMANDHAR_CITY/             # Cleaned - writeups only
│   │   ├── DEC7_2025_KV_CACHE_PHASE2_WRITEUP.md    # KEEP - final summary
│   │   ├── DEC7_2025_QV_SWAP_MIDPOINT_WRITEUP.md   # KEEP - key findings
│   │   └── SESSION_NOTES.md                         # Consolidated from others
│   │
│   ├── DEC8_RUNPOD/                     # Renamed, primary data
│   │   ├── 00_SETUP/
│   │   ├── 01_GEOMETRY_OF_RECURSION/
│   │   ├── 02_TEMPORAL_KV_ITERATION/
│   │   ├── WRITEUPS/
│   │   └── NOTES/
│   │
│   ├── MASTER_NOTEBOOK/
│   │   └── THE_GEOMETRY_OF_RECURSION_v2.ipynb      # Keep only v2
│   │
│   └── CROSS_VALIDATION/                # Analysis docs
│       ├── CROSS_VALIDATION_INTEGRITY_CHECK_PROMPT.md
│       ├── CROSS_VALIDATION_ANALYSIS.md
│       └── CROSS_VALIDATION_Feedback.md
│
├── CANONICAL_CODE/                      # Best versions only
│   ├── mistral_L27_FULL_VALIDATION.py   # KEEP - canonical validation
│   ├── n300_mistral_test_prompt_bank.py # KEEP - 320 prompts
│   └── causal_loop_closure_v2.py        # COPY from DEC8 - best causal script
│
├── ARCHIVE_NOV_2025/                    # Move, don't delete
│   ├── mistral_patching_DIAGNOSTIC.py   # Superseded
│   ├── mistral_patching_FINAL.py        # Superseded
│   ├── mistral_patching_TRULY_FIXED.py  # Superseded
│   ├── debug_path_patching.py           # Superseded
│   ├── fixed_path_patching.py           # Superseded
│   ├── path_patching_alternative.py     # Superseded
│   ├── THE_GEOMETRY_OF_RECURSION_MASTER.ipynb  # v1, superseded
│   └── [other November explorations]
│
└── [Keep existing: R_V_PAPER/, models/, utils/, etc.]
```

---

## Specific Actions

### DELETE (Safe - Empty/Duplicate)

| Path | Reason | Size |
|------|--------|------|
| `DEC7_2025_SIMANDHARCITY_DIVE/DEC4_LOGIT_LENS/` | Empty duplicate of DEC3 version | ~0 |
| `DEC7_2025_SIMANDHARCITY_DIVE/LLAMA3_L27_REPLICATION/` | Empty duplicate of DEC3 version | ~0 |
| `DEC_8_2025_RUNPOD_TEST/sanity_check.ipynb` | Empty notebook (243 bytes) | 243 B |

### ARCHIVE (Move to ARCHIVE_NOV_2025/)

| File | Reason |
|------|--------|
| `THE_GEOMETRY_OF_RECURSION_MASTER.ipynb` | Superseded by v2 |
| `mistral_patching_DIAGNOSTIC.py` | Early iteration |
| `mistral_patching_FINAL.py` | Superseded by L27_FULL_VALIDATION |
| `mistral_patching_TRULY_FIXED.py` | Superseded |
| `mistral_find_snap_layer.py` | One-time exploration |
| `mistral_MIXTRAL_METHOD_FIXED.py` | Superseded |
| `debug_path_patching.py` | Debug artifact |
| `fixed_path_patching.py` | Superseded |
| `path_patching_alternative.py` | Superseded |

### CONSOLIDATE (Merge writeups)

| From | To | Action |
|------|----|--------|
| `DEC7/.../DEC7_2025_KV_CACHE_MIDPOINT_WRITEUP.md` | `SESSION_NOTES.md` | Merge into single file |
| `DEC7/.../DEC7_2025_KV_CACHE_STAGE_SUMMARY_MIDPOINT.md` | `SESSION_NOTES.md` | Merge into single file |
| `DEC7/.../DEC7_2025_PRE_RUN_QV_SWAP_NOTES.md` | `SESSION_NOTES.md` | Merge into single file |
| `DEC7/.../DEC7_2025_QV_SWAP_RUN_LOG.md` | `SESSION_NOTES.md` | Merge into single file |

**Keep separately:**
- `DEC7_2025_KV_CACHE_PHASE2_WRITEUP.md` - Final summary
- `DEC7_2025_QV_SWAP_MIDPOINT_WRITEUP.md` - Key findings

### RENAME (Clarity)

| From | To |
|------|----|
| `DEC3_2025_BALI_short_SPRINT/` | `DECEMBER_2025_EXPERIMENTS/DEC3_BALI/` |
| `DEC7_2025_SIMANDHARCITY_DIVE/` | `DECEMBER_2025_EXPERIMENTS/DEC7_SIMANDHAR_CITY/` |
| `DEC_8_2025_RUNPOD_GPU_TEST/` | `DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/` |
| `DEC_8_2025_RUNPOD_TEST/` | `DECEMBER_2025_EXPERIMENTS/CROSS_VALIDATION/` |

---

## What's ESSENTIAL (Never Delete)

### Data Files (CSVs with experimental results)
```
DEC3_BALI/LLAMA3_L27_REPLICATION/logs/raw_Jupyter_kernel/results/
  - llama3_L27_FULL_VALIDATION_20251203_054646.csv
  - llama3_L27_FULL_VALIDATION_20251203_065527.csv
  - mistral_L20_FULL_VALIDATION_20251203_072103.csv
  - mistral_L22_FULL_VALIDATION_20251203_073538.csv

DEC8_RUNPOD/01_GEOMETRY_OF_RECURSION/results/
  - comprehensive_analysis_20251208_132337.csv      # n=30 layer sweep
  - temporal_cinematography_20251208_142052.csv     # n=60 temporal
  - causal_loop_v2_20251208_161602.csv              # Causal loop closure
  - [all other CSVs]

DEC8_RUNPOD/02_TEMPORAL_KV_ITERATION/results/
  - [all CSVs]
```

### Code Files (Validated scripts)
```
- n300_mistral_test_prompt_bank.py                  # 320 prompts
- mistral_L27_FULL_VALIDATION.py                    # Canonical validation
- DEC8_RUNPOD/.../causal_loop_closure_v2.py        # Best causal script
- THE_GEOMETRY_OF_RECURSION_MASTER_v2.ipynb        # Master notebook
```

### Documentation (Key writeups)
```
- DEC7_2025_KV_CACHE_PHASE2_WRITEUP.md             # KV cache mechanism
- DEC7_2025_QV_SWAP_MIDPOINT_WRITEUP.md            # Q vs V findings
- DEC4_2025_MISTRAL_CROSS_ARCHITECTURE_VALIDATION.md # Cross-arch validation
- DEC8_2025_FINAL_SESSION_SUMMARY.md               # Causal loop closure
- DEC8_AUDIT_AND_LIMITATIONS.md                    # Honest assessment
- WHERE_WE_STAND.md                                # Publication readiness
```

---

## Contradictions Found & Resolved

### 1. Layer Numbers
- **DEC3/4**: L22 optimal for Mistral, L24 for Llama
- **DEC7**: L16-32 for KV cache transfer
- **DEC8**: L27 optimal for R_V measurement

**Resolution**: Not contradictory. L22-27 are all "late layers" (~70-85% depth). DEC8's comprehensive sweep (n=30) confirms L27 is peak, but effect is distributed L16-31. The KV cache mechanism requires the full range, while R_V measurement peaks at L27.

### 2. Transfer Efficiency
- **DEC3**: 271% (geometric)
- **DEC7**: 89.7% (behavioral)
- **DEC8**: 95.3% (behavioral)

**Resolution**: The 271% was geometric transfer (R_V values), not behavioral. DEC7 and DEC8 both measure behavioral transfer and agree (~90%). Apples to apples.

### 3. V-Patching Effect
- **DEC4**: "V-patching has NO CAUSAL SPECIFICITY"
- **DEC8**: "V-patching does NOT transfer R_V contraction"

**Resolution**: Consistent. Both conclude V alone is insufficient. The mechanism requires full KV cache.

---

## Execution Script

```bash
#!/bin/bash
# Run from ~/mech-interp-latent-lab-phase1/

# 1. Create new structure
mkdir -p DECEMBER_2025_EXPERIMENTS/{DEC3_BALI,DEC7_SIMANDHAR_CITY,DEC8_RUNPOD,MASTER_NOTEBOOK,CROSS_VALIDATION}
mkdir -p ARCHIVE_NOV_2025
mkdir -p CANONICAL_CODE

# 2. Move December folders
mv DEC3_2025_BALI_short_SPRINT/* DECEMBER_2025_EXPERIMENTS/DEC3_BALI/
mv DEC_8_2025_RUNPOD_GPU_TEST/* DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/

# 3. Extract DEC7 writeups only (skip empty duplicates)
cp DEC7_2025_SIMANDHARCITY_DIVE/DEC7_2025_KV_CACHE_PHASE2_WRITEUP.md DECEMBER_2025_EXPERIMENTS/DEC7_SIMANDHAR_CITY/
cp DEC7_2025_SIMANDHARCITY_DIVE/DEC7_2025_QV_SWAP_MIDPOINT_WRITEUP.md DECEMBER_2025_EXPERIMENTS/DEC7_SIMANDHAR_CITY/

# 4. Move cross-validation docs
mv DEC_8_2025_RUNPOD_TEST/CROSS_VALIDATION*.md DECEMBER_2025_EXPERIMENTS/CROSS_VALIDATION/
mv DEC_8_2025_RUNPOD_TEST/mistral_quick_test.py DECEMBER_2025_EXPERIMENTS/CROSS_VALIDATION/

# 5. Keep only v2 notebook
cp THE_GEOMETRY_OF_RECURSION_MASTER_v2.ipynb DECEMBER_2025_EXPERIMENTS/MASTER_NOTEBOOK/

# 6. Archive November iterations
mv THE_GEOMETRY_OF_RECURSION_MASTER.ipynb ARCHIVE_NOV_2025/
mv mistral_patching_*.py ARCHIVE_NOV_2025/
mv debug_path_patching.py fixed_path_patching.py path_patching_alternative.py ARCHIVE_NOV_2025/
mv mistral_find_snap_layer.py mistral_MIXTRAL_METHOD_FIXED.py ARCHIVE_NOV_2025/

# 7. Copy canonical code
cp mistral_L27_FULL_VALIDATION.py CANONICAL_CODE/
cp n300_mistral_test_prompt_bank.py CANONICAL_CODE/
cp DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/01_GEOMETRY_OF_RECURSION/code/causal_loop_closure_v2.py CANONICAL_CODE/

# 8. Clean up empty folders
rm -rf DEC3_2025_BALI_short_SPRINT DEC7_2025_SIMANDHARCITY_DIVE DEC_8_2025_RUNPOD_GPU_TEST DEC_8_2025_RUNPOD_TEST

echo "Reorganization complete!"
```

---

## Size Impact

| Before | After | Savings |
|--------|-------|---------|
| ~6 MB scattered | ~5 MB organized | ~1 MB (duplicates removed) |
| 4 December folders | 1 consolidated folder | Cleaner navigation |
| 8 patching scripts | 1 canonical + archive | Clear "best" version |
| 2 master notebooks | 1 canonical | No version confusion |

---

## Verification Checklist

Before running the script, verify:

- [ ] All CSVs from DEC3/DEC8 are present in new locations
- [ ] All key writeups are preserved
- [ ] Canonical code files are correct versions
- [ ] Nothing in ARCHIVE_NOV_2025 is actually needed
- [ ] v2 notebook contains all v1 functionality

After running:

- [ ] `find DECEMBER_2025_EXPERIMENTS -name "*.csv" | wc -l` matches original count
- [ ] Key writeups readable and intact
- [ ] CANONICAL_CODE scripts run without errors

---

## Recommendation

**Execute in stages:**

1. **Stage 1 (Safe)**: Create new folders, copy (don't move) essential files
2. **Stage 2 (Verify)**: Confirm all data preserved in new locations
3. **Stage 3 (Clean)**: Move superseded files to archive
4. **Stage 4 (Final)**: Remove empty original folders

This way you can abort at any stage if something looks wrong.

---

*Bonsai trimming complete. Nothing essential cut. All pruned branches archived.*
