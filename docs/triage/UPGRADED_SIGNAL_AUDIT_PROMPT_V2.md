# Upgraded Comprehensive Signal Quality & Industry Standard Audit Prompt

**Version:** 2.1  
**Date:** 2026-02-05  
**Purpose:** Enhanced audit prompt incorporating lessons learned from initial audit

---

## Your Mission

You are auditing `mech-interp-latent-lab-phase1` for **signal quality** and **industry‑standard rigor**.

### Primary Goal
Reorganize the repo to privilege **high‑N, high‑signal, contract‑compliant** work.  
Everything else should be **archived** or **queued for ramp‑up**.

---

## Non‑Negotiables (Industry Standard)

### 1. R_V Metric Definition (CRITICAL - Enhanced Checks)

**MUST BE**: `R_V = PR_late / PR_early` (ratio across layers)

**Verification Steps:**
1. Check `src/metrics/rv.py` - must compute ratio (✅ canonical)
2. Check `rv_toolkit/rv_toolkit/metrics.py` - ⚠️ currently single-layer only
3. **Check all result JSON/CSV files** - verify R_V values are ratios:
   - R_V < 1.0 for recursive prompts (contraction)
   - R_V ≈ 1.0 for baseline prompts (neutral)
   - **Flag any R_V > 1.5** as likely single-layer PR mislabeled

**Common Violations:**
- Single-layer PR values (5-10 range) labeled as "rv"
- Values >1.0 for recursive prompts (should be <1.0)
- Missing early/late layer distinction

**Fix Priority:** CRITICAL - Contract violation invalidates results

---

### 2. Sample Size Thresholds (Enhanced)

**Minimum**: n ≥ 50 pairs/prompts per condition  
**Preferred**: n ≥ 80 for publication claims  
**Exploratory**: n ≥ 30 acceptable if clearly marked as exploratory

**Verification:**
- Check `summary.json` for `n_pairs` or `n_prompts`
- Check CSV files for actual row count (may differ from config)
- Flag experiments with n < 50 as "RAMP_UP" unless exploratory

**Exceptions:**
- Layer sweeps: n ≥ 20 per layer acceptable (many layers tested)
- Head ablations: n ≥ 30 acceptable (many heads tested)
- Smoke tests: n ≥ 5 acceptable (quick validation only)

---

### 3. Statistical Requirements (Enhanced)

**Required Statistics:**
- Cohen's d (effect size)
- p-value (significance)
- 95% CI (confidence interval)
- n (sample size)

**Multiple Comparisons:**
- If testing multiple conditions: Bonferroni or Holm-Bonferroni correction
- If testing multiple models: Report both corrected and uncorrected p-values
- Document correction method used

**Verification:**
- Check `summary.json` for all required stats
- Verify p-values are corrected if multiple tests
- Flag missing stats as "INCOMPLETE"

---

### 4. Control Requirements (Enhanced)

**Required Controls (by experiment type):**

**Causal Validation:**
- ✅ Baseline (no intervention)
- ✅ Random (content-specificity)
- ✅ Shuffled (structure-specificity)
- ✅ Wrong-layer (layer-specificity)

**Confound Validation:**
- ✅ Length-matched (token length control)
- ✅ Pseudo-recursive (recursion-like but not recursive)
- ✅ Pure repetition (kill switch - should NOT contract)

**Behavioral Validation:**
- ✅ Baseline generation
- ✅ Degeneracy gates (4-gram repeat, unique word ratio)
- ✅ Random KV (content control)

**Verification:**
- Check config for `include_controls` or control conditions
- Check summary.json for control statistics
- Flag missing controls as "INCOMPLETE"

---

### 5. Artifact Standards (Enhanced)

**Required Artifacts (Every Experiment):**

| Artifact | Required | Format | Purpose |
|----------|----------|--------|---------|
| `config.json` | ✅ YES | JSON | Exact config snapshot |
| `summary.json` | ✅ YES | JSON | Machine-readable summary with stats |
| `*_results.csv` or `*_pairs.csv` | ✅ YES | CSV | Per-sample data |
| `prompt_bank_version.txt` or `.json` | ✅ YES | Text/JSON | Prompt bank hash (reproducibility) |
| `hardware_info.json` | ✅ YES | JSON | GPU, CUDA, precision (reproducibility) |
| `report.md` | ⚠️ OPTIONAL | Markdown | Human-readable summary |
| `metadata.json` | ⚠️ OPTIONAL | JSON | Run metadata (git commit, timestamp) |

**hardware_info.json Format:**
```json
{
  "gpu_name": "NVIDIA L40S",
  "cuda_version": "12.1",
  "torch_version": "2.1.2",
  "torch_dtype": "float16",
  "device": "cuda",
  "python_version": "3.11.0"
}
```

**Verification:**
- Check run directory for all required artifacts
- Flag missing artifacts as "INCOMPLETE"
- Archive incomplete runs to `results/archive/incomplete/`

---

### 6. Prompt Bank Compliance (Enhanced)

**Source:** MUST use canonical `prompts/bank.json`

**Version Tracking:**
- MUST log prompt bank hash (SHA256, first 16 chars)
- Store in `prompt_bank_version.txt` or `prompt_bank_version.json`
- Include in `summary.json` as `prompt_bank_version`

**Verification:**
- Check for hardcoded prompts in pipeline code
- Check for prompt bank version in artifacts
- Flag missing version tracking as "INCOMPLETE"

---

## Deliverables (Enhanced Format)

### 1. KEEP_SIGNAL List

**Format:**
```
File Path | n | Stats | Controls | Artifacts | R_V Correct? | Why High-Signal
----------|---|-------|----------|-----------|--------------|----------------
results/.../summary.json | 45 | d=-3.56, p<1e-6 | ✅ 4 types | ✅ All present | ✅ Yes | Causal validation with perfect control separation
```

**Criteria for KEEP:**
- n ≥ 50 (or n ≥ 45 with strong effect and complete controls)
- ✅ Proper R_V ratio implementation (PR_late/PR_early)
- ✅ Complete controls (at least 3 types for causal claims)
- ✅ Complete artifacts (all required files present)
- ✅ Statistically significant (p < 0.01, corrected)
- ✅ Effect size |d| ≥ 0.5
- ✅ R_V values consistent with contraction (R_V < 1.0 for recursive)

**New Column:** "R_V Correct?" - Verify R_V values are ratios, not single-layer PR

---

### 2. RAMP_UP List

**Format:**
```
File Path | Current n | Target n | Missing | Config Changes | Priority
----------|-----------|----------|---------|---------------|----------
results/.../summary.json | 20 | 50 | wrong_layer, hardware_info | {"n_pairs": 50, "include_controls": ["wrong_layer"]} | HIGH
```

**Criteria for RAMP_UP:**
- Promising finding (effect present, even if small n)
- Missing controls or artifacts (fixable)
- Can reach industry standard with config changes
- Not superseded by better experiment
- Clear path to high-N

**New Column:** "Priority" - HIGH/MEDIUM/LOW based on scientific value

---

### 3. ARCHIVE_ONLY List

**Format:**
```
File Path | Reason | Evidence | Archive Location
----------|--------|----------|------------------
results/.../final_results.json | Contract violation | Single-layer PR (5.279) mislabeled as R_V | results/archive/contract_violations/
```

**Reasons (Enhanced):**
- **Contract violation**: Wrong R_V definition, missing artifacts, incorrect stats
- **Confound**: Missing critical controls, confounded design
- **Outdated**: Superseded by better experiment
- **Duplicate**: Same finding, lower quality
- **Dead-end**: No signal, no path forward
- **Incomplete**: Missing required artifacts, cannot reproduce

**New Column:** "Archive Location" - Where to move the file

---

### 4. Top 5 ROI Experiments

**Format:**
```
Rank | Experiment | Current State | Gap to Bridge | Config Path | Expected Outcome | Effort | Priority
-----|------------|---------------|---------------|-------------|------------------|--------|----------
1 | Multi-token R_V→behavior | n=40, weak correlation | n=80, token-by-token R_V | configs/phase3_bridge/... | Bridge R_V to L4 markers | 2 days | CRITICAL
```

**Criteria (Enhanced):**
- Directly advances causal bridge (R_V → behavior)
- Clear path to industry standard
- High scientific value
- Feasible with current infrastructure
- **Effort estimate** (days/weeks)
- **Priority** (CRITICAL/HIGH/MEDIUM)

**New Columns:** "Effort" and "Priority" for planning

---

### 5. Claims vs Data Audit

**Format:**
```
Claim Location | Claim | Data Location | Verification | Status | Action Required
---------------|-------|---------------|--------------|--------|-----------------
RECOVERED_GOLD/...md | "d=-3.56" | results/.../summary.json | ✅ Verified: d=-3.558 | VALID | None
results/.../final_results.json | "baseline_rv=5.279" | [CONTRACT VIOLATION] | ❌ Single-layer PR, not R_V | INVALID | Re-compute R_V ratio
```

**Status Types:**
- **VALID**: Claim matches data
- **INVALID**: Claim contradicts data or violates contract
- **UNTRACEABLE**: No data found to verify claim
- **UNCERTAIN**: Data exists but unclear/ambiguous

**New Column:** "Action Required" - What needs to be fixed

---

## Audit Process (Enhanced)

### Step 1: Scan All Results Directories
```bash
# Priority order
1. results/canonical/          # Priority 1: Check all
2. results/phase1_cross_architecture/  # Priority 1: Cross-model validation
3. results/phase3_bridge/     # Priority 1: R_V → behavior bridge
4. results/discovery/          # Priority 2: Check promising ones
5. results/archive/            # Skip (already archived)
```

### Step 2: Verify R_V Implementation (Enhanced)
1. Check `src/metrics/rv.py` - must compute `PR_late / PR_early` ✅
2. Check `rv_toolkit/rv_toolkit/metrics.py` - flag if single-layer only ⚠️
3. **Check all result JSON/CSV files** - verify R_V values:
   - Extract all "rv" or "baseline_rv" or "recursive_rv" values
   - Flag values >1.5 as likely single-layer PR
   - Flag values >1.0 for recursive prompts as contract violation
   - Verify early/late layer distinction in computation

### Step 3: Check Artifact Completeness (Enhanced)
For each result directory, verify:
- [ ] `config.json` exists
- [ ] `summary.json` exists with stats
- [ ] CSV with per-sample data exists
- [ ] `prompt_bank_version.*` exists
- [ ] `hardware_info.json` exists (NEW REQUIREMENT)
- [ ] `metadata.json` exists (optional but recommended)

**Missing Artifacts:** Archive to `results/archive/incomplete/`

### Step 4: Verify Statistical Claims (Enhanced)
For each `summary.json`:
- [ ] n ≥ 50 (or justified lower)
- [ ] Cohen's d reported
- [ ] p-value reported
- [ ] 95% CI reported (or can compute from mean, std, n)
- [ ] Multiple comparisons correction applied (if multiple tests)
- [ ] **R_V values are ratios, not single-layer PR** (NEW CHECK)

### Step 5: Check Controls (Enhanced)
For each experiment:
- [ ] Baseline condition present
- [ ] Appropriate controls for experiment type (see Control Requirements above)
- [ ] Control separation verified (different from main effect)
- [ ] Controls documented in config or summary

### Step 6: Cross-Reference Documentation (Enhanced)
- Check all markdown claims against actual data
- Flag untraceable claims
- Verify statistics match between docs and data
- **Check for contract violations** (single-layer PR mislabeled as R_V)

### Step 7: Check for Duplicates (NEW)
- Identify duplicate experiments (same config, different timestamps)
- Keep only highest-quality version (highest n, most complete artifacts)
- Archive others to `results/archive/duplicates/`

---

## Required Style (Enhanced)

- **Cite exact file paths** - Use full relative paths from repo root
- **Avoid speculation** - Mark uncertainty explicitly:
  - "⚠️ UNCERTAIN" - Data exists but unclear
  - "❓ NEEDS VERIFICATION" - Claim cannot be verified
  - "❌ CONTRACT VIOLATION" - Violates industry standard
- **Prioritize causal relevance** - Causal validation > exploratory analysis
- **Prioritize reproducibility** - Complete artifacts > incomplete artifacts
- **Flag contract violations immediately** - Don't wait for summary

---

## Scope Boundaries (Enhanced)

- **DO NOT** delete results
- **PREFER** marking and archiving over removing data
- **OUTCOME** should make repo "signal‑only" for future work
- **FOCUS** on results/ directory, not code (code audit separate)
- **ARCHIVE** incomplete/duplicate/violating results (don't delete)
- **DOCUMENT** archive locations and reasons

---

## Reference Files (Enhanced)

**Canonical Implementations:**
- `src/metrics/rv.py` - ✅ Correct R_V implementation (PR_late/PR_early)
- `configs/gold/02_causality.json` - Example industry-standard config
- `results/canonical/rv_l27_causal_validation/.../summary.json` - Example complete artifacts

**Previous Audits:**
- `STATISTICAL_AUDIT_REPORT.md` - Statistical validation
- `REPRODUCIBILITY_AUDIT_REPORT.md` - Reproducibility check
- `QUALITY_CONTROL_REPORT.md` - QC findings
- `COMPREHENSIVE_SIGNAL_AUDIT_REPORT_2026-02-05.md` - Previous comprehensive audit

**Known Issues:**
- `rv_toolkit/rv_toolkit/metrics.py` - ⚠️ Single-layer PR only (contract violation)
- `results/canonical/final_results.json` - ❌ Single-layer PR mislabeled as R_V

---

## Output Format (Enhanced)

Create a markdown report with sections:
1. Executive Summary (1-2 paragraphs, include contract violation count)
2. KEEP_SIGNAL (table with R_V correctness check)
3. RAMP_UP (table with priority column)
4. ARCHIVE_ONLY (table with archive location)
5. Top 5 ROI Experiments (detailed with effort/priority)
6. Claims vs Data Audit (table with action required)
7. Critical Gaps Summary (prioritized by severity)
8. Recommendations (immediate/short-term/long-term)
9. **Contract Violations Summary** (NEW - list all violations found)

---

## Key Improvements from V1.0

1. **Enhanced R_V Verification**: Check all result files for contract violations, not just implementations
2. **Hardware Info Requirement**: Now mandatory, not optional
3. **Priority/Effort Columns**: Added to RAMP_UP and ROI tables for planning
4. **Archive Locations**: Specify where to move archived files
5. **Action Required Column**: What needs to be fixed for each issue
6. **Contract Violations Summary**: Dedicated section for violations
7. **Enhanced Control Requirements**: Specific controls by experiment type
8. **Duplicate Detection**: New step to identify and archive duplicates
9. **R_V Correctness Check**: New column in KEEP_SIGNAL table
10. **Severity Levels**: CRITICAL/HIGH/MEDIUM/LOW for prioritization

---

**Begin audit now.**
