# Comprehensive Signal Quality & Industry Standard Audit Prompt

**Version:** 2.0  
**Date:** 2026-02-05  
**Purpose:** Full repository audit for signal quality and industry-standard rigor

---

## Your Mission

You are auditing `mech-interp-latent-lab-phase1` for **signal quality** and **industry‑standard rigor**.

### Primary Goal
Reorganize the repo to privilege **high‑N, high‑signal, contract‑compliant** work.  
Everything else should be **archived** or **queued for ramp‑up**.

---

## Non‑Negotiables (Industry Standard)

### 1. R_V Metric Definition (CRITICAL)
- **MUST BE**: `R_V = PR_late / PR_early` (ratio across layers)
- **NOT ACCEPTABLE**: Single-layer PR, absolute PR values, or any variant
- **Verification**: Check `src/metrics/rv.py` implementation
- **Flag**: Any results using single-layer PR as "R_V"

### 2. Sample Size Thresholds
- **Minimum**: n ≥ 50 pairs/prompts per condition
- **Preferred**: n ≥ 80 for publication claims
- **Flag**: Any experiment with n < 50 as "RAMP_UP" candidate

### 3. Statistical Requirements
- **Required**: Cohen's d, p-value, 95% CI
- **Multiple comparisons**: Bonferroni or Holm-Bonferroni correction
- **Flag**: Missing stats or uncorrected p-values

### 4. Control Requirements
- **Required controls**: random, shuffled, wrong_layer, orthogonal (where applicable)
- **Flag**: Missing controls or inadequate control separation

### 5. Artifact Standards
Every experiment MUST have:
- `config.json` - Exact config snapshot
- `summary.json` - Machine-readable summary with stats
- `*_results.csv` or `*_pairs.csv` - Per-sample data
- `prompt_bank_version.txt` or `.json` - Prompt bank hash
- `hardware_info.json` - GPU model, CUDA version, precision (FP16/BF16/FP32)

**Flag**: Missing artifacts as "INCOMPLETE"

### 6. Prompt Bank Compliance
- **Source**: MUST use canonical `prompts/bank.json`
- **Version tracking**: MUST log prompt bank hash
- **Flag**: Hardcoded prompts or missing version tracking

---

## Deliverables

### 1. KEEP_SIGNAL List
**Format:**
```
File Path | n | Stats | Controls | Artifacts | Why High-Signal
----------|---|-------|----------|-----------|----------------
results/canonical/rv_l27_causal_validation/.../summary.json | 45 | d=-3.56, p<1e-6 | random/shuffled/wrong_layer | ✅ All present | Causal validation with perfect control separation
```

**Criteria for KEEP:**
- n ≥ 50 (or n ≥ 45 with strong effect)
- Proper R_V ratio implementation
- Complete controls (at least 3 types)
- Complete artifacts
- Statistically significant (p < 0.01, corrected)
- Effect size |d| ≥ 0.5

### 2. RAMP_UP List
**Format:**
```
File Path | Current n | Target n | Missing Controls | Missing Artifacts | Config Changes Needed
----------|-----------|----------|------------------|-------------------|----------------------
results/discovery/.../summary.json | 20 | 50 | wrong_layer | hardware_info.json | {"n_pairs": 50, "include_controls": ["wrong_layer"]}
```

**Criteria for RAMP_UP:**
- Promising finding (effect present, even if small n)
- Missing controls or artifacts
- Can reach industry standard with config changes
- Not superseded by better experiment

### 3. ARCHIVE_ONLY List
**Format:**
```
File Path | Reason | Evidence
----------|--------|----------
results/session_2/.../results.json | Superseded | Duplicate of canonical/rv_l27_causal_validation with lower n
archive/old_experiments/... | Contract violation | Uses single-layer PR, not ratio
```

**Reasons:**
- **Confound**: Missing critical controls, confounded design
- **Outdated**: Superseded by better experiment
- **Duplicate**: Same finding, lower quality
- **Contract violation**: Wrong R_V definition, missing artifacts
- **Dead-end**: No signal, no path forward

### 4. Top 5 ROI Experiments
**Format:**
```
Rank | Experiment | Current State | Gap to Bridge | Config Path | Expected Outcome
-----|------------|---------------|---------------|-------------|------------------
1 | Multi-token R_V → behavior | Partial (n=40) | Need n=80, behavioral metrics | configs/phase3_bridge/.../multi_token_bridge.json | Bridge R_V contraction to L4 behavioral markers
```

**Criteria:**
- Directly advances causal bridge (R_V → behavior)
- Clear path to industry standard
- High scientific value
- Feasible with current infrastructure

### 5. Claims vs Data Audit
**Format:**
```
Claim Location | Claim | Data Location | Verification | Status
---------------|-------|--------------|--------------|--------
RECOVERED_GOLD/MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md | "Cohen's d = -3.56" | results/canonical/.../summary.json | ✅ Verified: d=-3.558 | VALID
docs/analysis/... | "R_V < 0.6 for all recursive prompts" | [NO DATA FOUND] | ❌ Cannot verify | UNTRACEABLE
```

**Flag:**
- Claims without traceable data
- Claims contradicting data
- Claims with wrong statistics (e.g., reporting single-layer PR as R_V)

---

## Audit Process

### Step 1: Scan All Results Directories
```bash
# Check structure
results/
├── canonical/          # Priority 1: Check all
├── discovery/          # Priority 2: Check promising ones
├── phase1_cross_architecture/  # Priority 1: Cross-model validation
├── phase3_bridge/     # Priority 1: R_V → behavior bridge
└── archive/            # Skip (already archived)
```

### Step 2: Verify R_V Implementation
- Check `src/metrics/rv.py` - must compute `PR_late / PR_early`
- Check `rv_toolkit/rv_toolkit/metrics.py` - flag if single-layer only
- Check all result CSVs - verify R_V values are ratios, not single PR

### Step 3: Check Artifact Completeness
For each result directory:
- [ ] `config.json` exists
- [ ] `summary.json` exists with stats
- [ ] CSV with per-sample data exists
- [ ] `prompt_bank_version.*` exists
- [ ] `hardware_info.json` exists (or documented in summary.json)

### Step 4: Verify Statistical Claims
For each `summary.json`:
- [ ] n ≥ 50 (or justified lower)
- [ ] Cohen's d reported
- [ ] p-value reported
- [ ] 95% CI reported (or can compute from mean, std, n)
- [ ] Multiple comparisons correction applied (if multiple tests)

### Step 5: Check Controls
For each experiment:
- [ ] Baseline condition present
- [ ] At least 2 control conditions (random, shuffled, wrong_layer, etc.)
- [ ] Control separation verified (different from main effect)

### Step 6: Cross-Reference Documentation
- Check all markdown claims against actual data
- Flag untraceable claims
- Verify statistics match between docs and data

---

## Required Style

- **Cite exact file paths** - Use full relative paths from repo root
- **Avoid speculation** - Mark uncertainty explicitly with "⚠️ UNCERTAIN" or "❓ NEEDS VERIFICATION"
- **Prioritize causal relevance** - Causal validation > exploratory analysis
- **Prioritize reproducibility** - Complete artifacts > incomplete artifacts

---

## Scope Boundaries

- **DO NOT** delete results
- **PREFER** marking and archiving over removing data
- **OUTCOME** should make repo "signal‑only" for future work
- **FOCUS** on results/ directory, not code (code audit separate)

---

## Reference Files

- `src/metrics/rv.py` - Canonical R_V implementation (PR_late/PR_early)
- `configs/gold/02_causality.json` - Example industry-standard config
- `results/canonical/rv_l27_causal_validation/.../summary.json` - Example complete artifacts
- `RECOVERED_GOLD/MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md` - Example validated findings
- `STATISTICAL_AUDIT_REPORT.md` - Previous statistical validation
- `REPRODUCIBILITY_AUDIT_REPORT.md` - Previous reproducibility check
- `QUALITY_CONTROL_REPORT.md` - Previous QC findings

---

## Output Format

Create a markdown report with sections:
1. Executive Summary (1-2 paragraphs)
2. KEEP_SIGNAL (table)
3. RAMP_UP (table)
4. ARCHIVE_ONLY (table)
5. Top 5 ROI Experiments (detailed)
6. Claims vs Data Audit (table)
7. Critical Gaps Summary
8. Recommendations

---

**Begin audit now.**
