# Documentation Redundancy Analysis & Consolidation Plan

**Project:** mech-interp-latent-lab-phase1  
**Analysis Date:** 2026-02-05  
**Total Documents Analyzed:** ~400+ markdown files  

---

## Executive Summary

The repository has **significant documentation redundancy** with ~35% of documents containing overlapping or superseded information. The docs/ folder alone contains **304 markdown files** with heavy sprawl in misc/ (136 files) and analysis/ (34 files).

**Key Finding:** Most redundancy stems from:
1. Multiple audit reports covering the same findings
2. Session logs that should be archived
3. Orphaned docs in docs/misc/
4. Duplicate status/summary documents
5. Root-level .md files that should be in docs/

---

## Category 1: REDUNDANT DOCUMENTS (Merge/Delete Candidates)

### 1.1 Audit Report Redundancy (5 docs → 1)

| Document | Location | Redundancy | Action |
|----------|----------|------------|--------|
| AUDIT_REPORT_2026-02-05.md | Root | Feb 5 audit | **MERGE INTO** COMPREHENSIVE_SIGNAL_AUDIT |
| COMPREHENSIVE_SIGNAL_AUDIT_REPORT_2026-02-05.md | Root | Feb 5 audit | **KEEP** (most comprehensive) |
| docs/analysis/AUDIT.md | docs/analysis/ | Jan 11 audit | **ARCHIVE** (superseded) |
| STATISTICAL_AUDIT_REPORT.md | Root | Statistical focus | **MERGE** unique findings into COMPREHENSIVE |
| STATISTICAL_AUDIT_EXECUTIVE_SUMMARY.md | Root | Summary only | **DELETE** (duplicate of main report) |

**Rationale:** All Feb 5 audits cover the same codebase state. Keep the most comprehensive (COMPREHENSIVE_SIGNAL_AUDIT) and merge unique insights from others.

### 1.2 Status Document Redundancy (8 docs → 2)

| Document | Location | Status | Action |
|----------|----------|--------|--------|
| docs/status/STATUS.md | docs/status/ | Current | **KEEP** (authoritative) |
| docs/status/STATUS_JAN15_2025.md | docs/status/ | Historical | **ARCHIVE** to docs/archive/ |
| docs/experiments/EXPERIMENT_STATUS.md | docs/experiments/ | Circuit Hunt V2 | **MERGE** into STATUS.md |
| docs/status/V3_STATUS.md | docs/status/ | Old version | **DELETE** |
| docs/status/STAGE_1_COMPLETE_SUMMARY.md | docs/status/ | Stage 1 | **MERGE** into STATUS.md |
| docs/status/STAGE_2_FINAL_REPORT.md | docs/status/ | Stage 2 | **MERGE** into STATUS.md |
| docs/status/PHASE1_SUMMARY.md | docs/status/ | Phase 1 | **MERGE** into STATUS.md |
| docs/status/RESEARCH_PROGRESS_SUMMARY.md | docs/status/ | General | **DELETE** (redundant) |

### 1.3 Summary/Synthesis Redundancy (6 docs → 2)

| Document | Location | Content | Action |
|----------|----------|---------|--------|
| docs/misc/FINAL_SYNTHESIS_AND_INSIGHTS.md | docs/misc/ | High-level | **KEEP** (narrative synthesis) |
| docs/status/FINAL_REPORT_DEC16.md | docs/status/ | Dec 16 report | **ARCHIVE** (historical) |
| docs/status/FINAL_REPORT_DEC19.md | docs/status/ | Dec 19 report | **ARCHIVE** (historical) |
| docs/status/COMPREHENSIVE_RESEARCH_SUMMARY.md | docs/status/ | Research summary | **MERGE** into FINAL_SYNTHESIS |
| docs/status/COMPREHENSIVE_INVESTIGATION_REPORT.md | docs/status/ | Investigation | **MERGE** into FINAL_SYNTHESIS |
| docs/misc/MY_HONEST_REFLECTION.md | docs/misc/ | Personal | **ARCHIVE** (not project-critical) |

### 1.4 Architecture Documentation (4 docs → 2)

| Document | Location | Content | Action |
|----------|----------|---------|--------|
| ARCHITECTURE_RESTRUCTURE_PLAN.md | Root | Full plan | **KEEP** (canonical) |
| ARCHITECTURE_EXECUTIVE_SUMMARY.md | Root | Summary | **MERGE** into RESTRUCTURE_PLAN |
| ARCHITECTURE_VISUAL_GUIDE.md | Root | Visual | **KEEP** (complementary) |
| IMPLEMENTATION_CHECKLIST.md | Root | Checklist | **MERGE** into RESTRUCTURE_PLAN |

### 1.5 Quality/Audit Documentation (4 docs → 2)

| Document | Location | Content | Action |
|----------|----------|---------|--------|
| QUALITY_CONTROL_REPORT.md | Root | QC findings | **KEEP** (current) |
| README_AUDIT_RESULTS.md | Root | README audit | **MERGE** into QUALITY_CONTROL |
| REPRODUCIBILITY_AUDIT_REPORT.md | Root | Reproducibility | **MERGE** into COMPREHENSIVE_SIGNAL_AUDIT |
| PUBLICATION_BLOCKERS_STATUS.md | Root | Blockers | **MERGE** into QUALITY_CONTROL |

---

## Category 2: OUTDATED DOCS (Archive/Delete)

### 2.1 Session Logs (23 docs → Archive All)

All session logs in `docs/sessions/` are historical working notes. **Action: Move all to docs/archive/sessions/**

| Document | Date | Reason |
|----------|------|--------|
| NOV_19_FULL_SESSION_LOG.md | Nov 2024 | Historical |
| DEC12_2024_DEEP_ANALYSIS_SESSION.md | Dec 2024 | Historical |
| DEC13_SESSION_LOG.md | Dec 2024 | Historical |
| SESSION_SUMMARY_DEC17.md | Dec 2024 | Historical |
| SESSION_SUMMARY_DEC19.md | Dec 2024 | Historical |
| JAN11_2025_SESSION_SUMMARY.md | Jan 2025 | Historical |
| AGENT_HANDOFF_JAN15_2025.md | Jan 2025 | Historical |
| 2026-01-25_CONSOLIDATED_AUDIT_ACTION_PLAN.md | Jan 2026 | **EXTRACT** action items, then archive |
| ... (15 more) | Various | Historical |

**Exception:** If any session contains **untranscribed findings**, extract those to docs/findings/ before archiving.

### 2.2 Deprecated Analysis Documents (docs/misc/)

The `docs/misc/` folder has **136 files** - many are deprecated. **Action: Review and archive ~80%**

| Pattern | Count | Action |
|---------|-------|--------|
| C2_*, C1_* results | ~15 | **ARCHIVE** to docs/archive/c2_results/ |
| ITERATION_V* | ~10 | **ARCHIVE** (old iterations) |
| DEC12_*, DEC13_* | ~20 | **ARCHIVE** (December working notes) |
| STEERING_* fix docs | ~15 | **MERGE** into single STEERING_FIXES.md |
| GPU_SETUP_* | ~5 | **CONSOLIDATE** into SETUP_GUIDE.md |
| RUNPOD_* | ~8 | **CONSOLIDATE** into RUNPOD_GUIDE.md |
| CIRCUIT_HUNT_V2_* | ~6 | **MERGE** into CIRCUIT_HUNT_V2_SUMMARY.md |
| THE_* essays | ~8 | **KEEP** (philosophical content) |

### 2.3 Superseded Experimental Docs

| Document | Location | Reason | Action |
|----------|----------|--------|--------|
| docs/experiments/VERIFICATION_EXPERIMENT_PLAN.md | docs/experiments/ | Plan outdated | **ARCHIVE** |
| docs/experiments/VERIFICATION_EXPERIMENT_ACTUAL_OUTPUTS.md | docs/experiments/ | Old outputs | **ARCHIVE** |
| docs/experiments/PRIORITY_EXPERIMENTS_RUNNING.md | docs/experiments/ | Superseded | **DELETE** |
| docs/experiments/STAGE_1_SMOKE_TEST_*.md | docs/experiments/ | Completed | **ARCHIVE** |

### 2.4 Root-Level Duplicates

| Document | Root Duplicate? | Action |
|----------|-----------------|--------|
| BEHAVIOR_TRANSFER_ANALYSIS.md | Root + docs/analysis/ | **DELETE** root version |
| 20_MINUTE_REPRODUCIBILITY_PROTOCOL.md | Root | **MOVE** to docs/ |
| AGENT_PROMPT_GOLD_STANDARD.md | Root + docs/handoffs/ | **DELETE** root version |

---

## Category 3: ORPHANED DOCS (No Clear Purpose)

### 3.1 Triage Documents (4 docs → Review)

All in `docs/triage/` - appear to be prompt templates, not documentation.

| Document | Action |
|----------|--------|
| AUDIT_PROMPT.md | **MOVE** to prompts/ or DELETE |
| COMPREHENSIVE_SIGNAL_AUDIT_PROMPT.md | **MOVE** to prompts/ or DELETE |
| UPGRADED_SIGNAL_AUDIT_PROMPT_V2.md | **MOVE** to prompts/ or DELETE |
| REPO_SIGNAL_REORG.md | **MERGE** into RESTRUCTURE_PLAN |

### 3.2 Duplicate Gold Standard Docs

| Document | Location | Duplicate Of | Action |
|----------|----------|--------------|--------|
| docs/misc/GOLD_STANDARD_SUITE.md | docs/misc/ | docs/standards/INDEX.md | **MERGE** into standards/ |
| docs/misc/GOLD_STANDARD_RESEARCH_DIRECTIVE.md | docs/misc/ | AGENT_PROMPT_GOLD_STANDARD | **DELETE** |

### 3.3 Agent Review Responses (12 docs → Archive)

All in `agent_reviews/responses/` - these are agent outputs, not project docs.

**Action:** Move entire `agent_reviews/` folder to `docs/archive/agent_reviews/`

---

## Category 4: KEY DOCS TO PRESERVE (Critical)

### 4.1 Canonical Results (RECOVERED_GOLD/)

| Document | Why Critical |
|----------|--------------|
| MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md | **Crown jewel** - n=45 causal validation |
| PHASE_2_CIRCUIT_MAPPING_COMPLETE.md | Circuit mapping results |
| BREAKTHROUGH_BEHAVIOR_TRANSFER.md | Key finding documentation |
| GROUND_TRUTH_ASSESSMENT.md | Honest assessment |
| GRAND_UNIFIED_TEST_RESULTS.md | Unified patching results |
| HONEST_ASSESSMENT_PUBLICATION_REALITY.md | Publication assessment |

**Action:** KEEP all in RECOVERED_GOLD/. These are the validated, publication-critical documents.

### 4.2 Standards & Contracts (docs/standards/)

| Document | Why Critical |
|----------|--------------|
| MEASUREMENT_CONTRACT.md | **LOCKED** - canonical metric definitions |
| RULES_V2.md | Project rules |
| MIB_GAP_ANALYSIS.md | Gap analysis |
| INDEX.md | Standards index |

**Action:** KEEP all. These define project standards.

### 4.3 Core Documentation (Root)

| Document | Why Critical | Action |
|----------|--------------|--------|
| README.md | Project entry point | **KEEP** |
| CANONICAL_EXPERIMENTS.md | Experiment reference | **KEEP** |
| REPOSITORY_DISSECTION_COMPLETE.md | Repo analysis | **KEEP** |

### 4.4 Current Configuration Docs

| Document | Why Critical |
|----------|--------------|
| CLEANUP_PROPOSAL.md | Current cleanup plan |
| ARCHITECTURE_RESTRUCTURE_PLAN.md | Architecture target |
| UNIFIED_AUDITOR_INTEGRATION.md | Auditor specs |

### 4.5 Key Analysis Documents (docs/analysis/)

| Document | Why Critical |
|----------|--------------|
| SURGICAL_SWEEP_DEEP_ANALYSIS.md | Core surgical sweep findings |
| H1_CRITICAL_ANALYSIS.md | H1 analysis |
| GROUND_TRUTH_ASSESSMENT.md | Ground truth |
| HONEST_ASSESSMENT_PUBLICATION_REALITY.md | Publication reality |

---

## Consolidation Plan

### Phase 1: Archive Historical Sessions (30 min)

```bash
# Create archive structure
mkdir -p docs/archive/{sessions,analysis,experiments,misc}

# Move all session logs
mv docs/sessions/* docs/archive/sessions/

# Move outdated analysis
mv docs/analysis/AUDIT.md docs/archive/analysis/
mv docs/analysis/MY_HONEST_ASSESSMENT.md docs/archive/analysis/

# Move old experiment docs
mv docs/experiments/VERIFICATION_EXPERIMENT_* docs/archive/experiments/
mv docs/experiments/STAGE_1_SMOKE_TEST_* docs/archive/experiments/
mv docs/experiments/PRIORITY_EXPERIMENTS_RUNNING.md docs/archive/experiments/
```

### Phase 2: Merge Redundant Audits (45 min)

1. **Read** all Feb 5 audit reports
2. **Identify** unique content in each
3. **Create** CONSOLIDATED_AUDIT_REPORT.md incorporating all unique findings
4. **Delete** redundant individual reports (or move to docs/archive/audits/)

### Phase 3: Consolidate misc/ Sprawl (2 hours)

1. **Review** all 136 files in docs/misc/
2. **Categorize**:
   - Philosophical/THE_* essays → KEEP in docs/misc/
   - C2/C1 results → docs/archive/c2_results/
   - ITERATION_V* → docs/archive/iterations/
   - GPU/RUNPOD setup → docs/setup/ (consolidate)
   - STEERING_* → docs/steering/ (consolidate)
   - CIRCUIT_HUNT_* → docs/circuits/ (consolidate)

### Phase 4: Clean Root Level (30 min)

```bash
# Move docs to docs/
mv 20_MINUTE_REPRODUCIBILITY_PROTOCOL.md docs/
mv CLEANUP_PROPOSAL.md docs/
mv AGENT_ONBOARDING.md docs/

# Delete duplicates
rm BEHAVIOR_TRANSFER_ANALYSIS.md  # duplicate of docs/analysis/
rm AGENT_PROMPT_GOLD_STANDARD.md   # duplicate of docs/handoffs/

# Archive old reports
mkdir -p docs/archive/audits
mv AUDIT_REPORT_2026-02-05.md docs/archive/audits/  # merged into comprehensive
mv STATISTICAL_AUDIT_EXECUTIVE_SUMMARY.md docs/archive/audits/
```

### Phase 5: Consolidate Status Docs (30 min)

1. **Update** docs/status/STATUS.md with current state
2. **Merge** content from:
   - EXPERIMENT_STATUS.md
   - STAGE_1_COMPLETE_SUMMARY.md
   - STAGE_2_FINAL_REPORT.md
   - PHASE1_SUMMARY.md
3. **Archive** the merged source documents

### Phase 6: Organize Triage & Agent Reviews (15 min)

```bash
# Move triage prompts to prompts/
mv docs/triage/*.md prompts/auditor_prompts/
rmdir docs/triage/

# Move agent reviews to archive
mv agent_reviews docs/archive/
```

---

## Target Structure After Consolidation

```
mech-interp-latent-lab-phase1/
├── README.md                          # KEEP - entry point
├── CANONICAL_EXPERIMENTS.md           # KEEP - experiment reference
├── REPOSITORY_DISSECTION_COMPLETE.md  # KEEP - repo analysis
├── ARCHITECTURE_RESTRUCTURE_PLAN.md   # KEEP - architecture plan
│
├── docs/
│   ├── README.md                      # Docs index
│   ├── CONSOLIDATED_AUDIT_REPORT.md   # MERGED - single audit
│   ├── CURRENT_STATUS.md              # RENAME from STATUS.md
│   ├── FINAL_SYNTHESIS.md             # MERGED synthesis
│   │
│   ├── analysis/                      # Active analysis (15 files, down from 34)
│   │   ├── SURGICAL_SWEEP_DEEP_ANALYSIS.md
│   │   ├── H1_CRITICAL_ANALYSIS.md
│   │   ├── GROUND_TRUTH_ASSESSMENT.md
│   │   └── ... (12 more)
│   │
│   ├── standards/                     # Standards (4 files)
│   │   ├── MEASUREMENT_CONTRACT.md
│   │   ├── RULES_V2.md
│   │   └── ...
│   │
│   ├── experiments/                   # Active experiments (down from 70+)
│   │   ├── GRAND_UNIFIED_TEST_RESULTS.md
│   │   └── ... (recent only)
│   │
│   ├── findings/                      # Research findings
│   │   └── R_V_BEHAVIORAL_DISSOCIATION.md
│   │
│   ├── setup/                         # NEW - consolidated setup docs
│   │   ├── GPU_SETUP.md
│   │   ├── RUNPOD_GUIDE.md
│   │   └── REPRODUCIBILITY_PROTOCOL.md
│   │
│   ├── steering/                      # NEW - consolidated steering docs
│   │   └── STEERING_FIXES.md
│   │
│   ├── circuits/                      # NEW - consolidated circuit docs
│   │   └── CIRCUIT_HUNT_V2_SUMMARY.md
│   │
│   ├── misc/                          # Reduced from 136 to ~30
│   │   ├── THE_*.md                   # Philosophical essays
│   │   └── ... (selected keepers)
│   │
│   └── archive/                       # NEW - archived docs
│       ├── sessions/                  # All session logs
│       ├── analysis/                  # Superseded analysis
│       ├── experiments/               # Old experiment docs
│       ├── audits/                    # Merged audit reports
│       ├── agent_reviews/             # Agent review responses
│       └── misc/                      # Old misc docs
│
└── RECOVERED_GOLD/                    # KEEP - validated results
    ├── MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md
    └── ... (all preserved)
```

---

## Size Impact

| Category | Current | Target | Savings |
|----------|---------|--------|---------|
| docs/analysis/ | 34 files | 15 files | 19 archived |
| docs/sessions/ | 23 files | 0 files | 23 archived |
| docs/misc/ | 136 files | 30 files | 106 archived |
| docs/experiments/ | 70+ files | 20 files | 50 archived |
| Root .md files | 35 files | 10 files | 25 moved/deleted |
| agent_reviews/ | 12 files | 0 files | 12 archived |
| docs/status/ | 25 files | 5 files | 20 merged/archived |
| **TOTAL** | **~330 files** | **~80 files** | **~250 archived** |

---

## Recommended Priority

### High Priority (Do First)
1. Archive session logs (23 files) - clear historical records
2. Merge Feb 5 audit reports (5 files → 1) - reduce confusion
3. Move agent_reviews to archive (12 files) - not project docs

### Medium Priority (Do Second)
4. Consolidate misc/ sprawl (136 → 30 files) - biggest cleanup
5. Clean root-level duplicates (25 files → move/delete)
6. Merge status documents (20 → 5 files)

### Low Priority (Do Last)
7. Archive old experiment docs (50 files)
8. Consolidate analysis docs (34 → 15 files)
9. Final review and cleanup

---

## Appendix: Detailed File Lists

### A.1 Duplicate Content Mapping

```
AUDIT_REPORT_2026-02-05.md
└── 85% overlap with COMPREHENSIVE_SIGNAL_AUDIT_REPORT_2026-02-05.md

STATISTICAL_AUDIT_REPORT.md
└── 70% overlap with COMPREHENSIVE_SIGNAL_AUDIT (stats section)

docs/status/STATUS.md + EXPERIMENT_STATUS.md + PHASE1_SUMMARY.md
└── 60% overlap - all describe current project state

docs/misc/GOLD_STANDARD_SUITE.md + docs/standards/INDEX.md
└── 80% overlap - both document gold standard pipelines
```

### A.2 Orphaned Documents (No Clear Purpose)

- docs/triage/REPO_SIGNAL_REORG.md - unclear relationship to restructure plan
- docs/misc/META_INDEX.md - superseded by docs/misc/MASTER_INDEX.md
- docs/misc/CONFIG_CATEGORIZATION.md - outdated config analysis
- docs/misc/RESULTS_CATEGORIZATION.md - outdated results analysis

### A.3 Superseded by Code

- docs/misc/prompt_compatibility_scorer.py docs → now in src/utils/
- docs/misc/recursion_prompt_generator.py docs → now in src/utils/
- docs/misc/PROMPT_BANK_SEALED.md → now prompts/bank.json

---

**Analysis Complete**  
**Recommendation:** Execute Phase 1-3 immediately for maximum clarity gain with minimum effort.
