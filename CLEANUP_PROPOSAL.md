# Mech-Interp Latent Lab Phase 1 - Cleanup Proposal

**Current Size**: 356 MB | **Files**: ~5,500 | **Python**: 133 root + src/ | **MD**: 275 root

## Executive Summary

6 agents analyzed the repo. Here's what's **high signal** vs **noise**:

---

## PROTECTED FILES (DO NOT DELETE)

### Critical Code
```
CANONICAL_CODE/
├── n300_mistral_test_prompt_bank.py   # 320 prompts, sealed
├── mistral_L27_FULL_VALIDATION.py     # n=45 causal validation
└── causal_loop_closure_v2.py          # Loop closure test

src/metrics/
├── rv.py                              # THE R_V metric
├── behavior_strict.py                 # Behavioral validation
└── baseline_suite.py                  # Nanda-standard metrics

src/core/
├── hooks.py                           # V-projection capture
└── patching.py                        # Activation patching
```

### Critical Results (n>50)
```
results/canonical/c2_measurement_suite/     # n=1,141 (LARGEST)
results/canonical/rv_l27_causal_validation/ # n=45 pairs
results/phase1_mechanism/                   # n=3,785 MLP sufficiency
results/discovery/path_patching/            # n=43,200 exploratory
```

### Critical Documents
```
MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md   # Cohen's d = -3.56
REPOSITORY_DISSECTION_COMPLETE.md           # Repo analysis
HONEST_ASSESSMENT_PUBLICATION_REALITY.md    # Self-assessment
R_V_PAPER/research/                         # Paper materials
```

---

## SAFE TO DELETE (~50MB savings)

### 1. Boneyard (8.1 MB) - DELETE ENTIRE DIRECTORY
```bash
# First preserve ONE file with unique narrative:
cp ~/mech-interp-latent-lab-phase1/boneyard/DEC_9_EMERGENCY_BACKUP/OFFICIAL_DEC3_9_COMPREHENSIVE_REPORT.md \
   ~/mech-interp-latent-lab-phase1/R_V_PAPER/research/DEC3_9_DISCOVERY_NARRATIVE.md

# Then delete:
rm -rf ~/mech-interp-latent-lab-phase1/boneyard/
```
**Rationale**: 228 files, all superseded by canonical results. No unique data.

### 2. R_V_PAPER/code/ Dead Scripts (17 of 18 files)
Keep ONLY: `VALIDATED_mistral7b_layer27_activation_patching.py` (if exists)

Actually checking... it doesn't exist. The validated code is in CANONICAL_CODE/.
```bash
# Check what's actually in R_V_PAPER/code/
ls ~/mech-interp-latent-lab-phase1/R_V_PAPER/code/
```

**Files to DELETE from R_V_PAPER/code/**:
- mistral_patching_FINAL.py (uses Layer 21, wrong)
- mistral_patching_TRULY_FIXED.py
- mistral_patching_DIAGNOSTIC.py
- mistral_patching_FIXED_FINAL.py
- mistral_MIXTRAL_METHOD_FIXED.py
- mistral_EXACT_MIXTRAL_METHOD.py
- debug_path_patching.py
- fixed_path_patching.py
- All other iterative development artifacts

### 3. Root-Level Python Duplicates (~130 files to archive)
These are one-off scripts that belong in archive:

**DELETE (duplicates of CANONICAL_CODE/)**:
- `n300_mistral_test_prompt_bank.py` (root duplicate)
- `mistral_L27_FULL_VALIDATION.py` (root duplicate)

**ARCHIVE (move to `archive/scripts/`)**:
- All `phase*_*.py` files (15+ files)
- All `experiment_*.py` files (20+ files)
- All `test_*.py` files (10+ files)
- All `analyze_*.py` files (10+ files)
- All `validate_*.py` files (5+ files)
- All `reproduce_*.py` files (10+ files)

### 4. Root-Level MD Sprawl (275 files!)
**Keep at root** (essential):
- README.md
- MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md
- REPOSITORY_DISSECTION_COMPLETE.md
- HONEST_ASSESSMENT_PUBLICATION_REALITY.md

**Move to `docs/`** (everything else - ~270 files)

### 5. src/pipelines/ Cleanup
**DELETE orphaned archive files** (10 files):
- archive/anthropic_level_investigation.py
- archive/causal_mechanism_hunt.py
- archive/comprehensive_circuit_analysis.py
- archive/h31_ablation_causal.py
- archive/h31_investigation.py
- archive/hysteresis_patching.py
- archive/l27_deep_dive.py
- archive/prompt_bank_audit.py
- archive/unified_layer_map.py
- archive/p10_advanced_steering/ (entire dir)
- archive/gemini_config_generator/ (entire dir)

**DELETE root-level duplicates in src/pipelines/**:
- c2_rv_measurement.py (use discovery/ version)
- kv_mechanism.py
- logit_lens_analysis.py
- mlp_sufficiency_test.py
- mlp_vproj_combined_sufficiency_test.py
- vproj_patching_analysis.py

### 6. Pycache & System Files
```bash
find ~/mech-interp-latent-lab-phase1 -type d -name __pycache__ -exec rm -rf {} +
find ~/mech-interp-latent-lab-phase1 -name .DS_Store -delete
find ~/mech-interp-latent-lab-phase1 -name "*.pyc" -delete
```

---

## PROPOSED NEW STRUCTURE

```
mech-interp-latent-lab-phase1/
├── README.md
├── requirements.txt (create if missing)
├── CANONICAL_CODE/                    # Gold standard validated code
│   ├── n300_mistral_test_prompt_bank.py
│   ├── mistral_L27_FULL_VALIDATION.py
│   └── causal_loop_closure_v2.py
├── src/
│   ├── metrics/                       # Core R_V metric
│   ├── core/                          # Patching infrastructure
│   └── pipelines/
│       ├── canonical/                 # Production pipelines
│       └── discovery/                 # Research pipelines
├── results/
│   ├── canonical/                     # Publication-ready data
│   ├── phase1_mechanism/              # Mechanism validation
│   └── phase2_generalization/         # Cross-model results
├── R_V_PAPER/
│   ├── research/                      # Paper documents
│   ├── csv_files/                     # Core data files
│   └── results/                       # Mixtral etc.
├── configs/                           # Experiment configs
├── docs/                              # All other documentation
└── archive/                           # Old scripts (not boneyard)
```

---

## EXECUTION PLAN

### Phase 1: Backup (REQUIRED)
```bash
tar -czf ~/mech-interp-backup-$(date +%Y%m%d).tar.gz \
  ~/mech-interp-latent-lab-phase1/
```

### Phase 2: Safe Deletions
```bash
# Pycache and system files
find ~/mech-interp-latent-lab-phase1 -type d -name __pycache__ -exec rm -rf {} +
find ~/mech-interp-latent-lab-phase1 -name .DS_Store -delete

# Boneyard (after preserving narrative)
cp ~/mech-interp-latent-lab-phase1/boneyard/DEC_9_EMERGENCY_BACKUP/OFFICIAL_DEC3_9_COMPREHENSIVE_REPORT.md \
   ~/mech-interp-latent-lab-phase1/R_V_PAPER/research/DEC3_9_DISCOVERY_NARRATIVE.md
rm -rf ~/mech-interp-latent-lab-phase1/boneyard/
```

### Phase 3: Reorganization
```bash
# Create archive structure
mkdir -p ~/mech-interp-latent-lab-phase1/archive/scripts
mkdir -p ~/mech-interp-latent-lab-phase1/docs/{sessions,experiments,status}

# Move root Python scripts to archive
# (manual review recommended for each)

# Move MD files to docs/
# (manual review recommended)
```

---

## SIZE IMPACT ESTIMATE

| Action | Savings |
|--------|---------|
| Delete boneyard/ | 8.1 MB |
| Delete pycache | ~5 MB |
| Delete R_V_PAPER/code dead scripts | ~1 MB |
| Delete src/pipelines orphans | ~0.5 MB |
| Archive root scripts (no delete) | 0 MB |
| **TOTAL DELETION** | **~15 MB** |

*Note: Most savings come from reorganization, not deletion. The goal is clarity, not size.*

---

## HIGH-N VALIDATED DATASETS (PRESERVE)

| Dataset | n | Location | Status |
|---------|---|----------|--------|
| C2 Measurement Suite | 1,141 | results/canonical/c2_measurement_suite/ | KEEP |
| Path Patching | 43,200 | results/discovery/path_patching/ | KEEP |
| MLP Sufficiency | 3,785 | results/phase1_mechanism/ | KEEP |
| Causal Validation | 45 | results/canonical/rv_l27_causal_validation/ | KEEP - PAPER CRITICAL |
| Prompt Bank | 320 | CANONICAL_CODE/n300_mistral_test_prompt_bank.py | KEEP |

---

## APPROVAL REQUIRED

Before executing, confirm:
1. [ ] Backup created
2. [ ] OK to delete boneyard/ (8.1 MB, 228 files)
3. [ ] OK to delete R_V_PAPER/code/ dead scripts
4. [ ] OK to reorganize root files into docs/ and archive/

**Reply with "APPROVED" to proceed with Phase 2 deletions.**
