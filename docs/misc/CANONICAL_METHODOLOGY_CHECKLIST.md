# Canonical Methodology Checklist v1.0

**Date:** January 10, 2026  
**Source:** Synthesized from README.md, GOLD_STANDARD_RESEARCH_DIRECTIVE.md, INDUSTRY_GRADE_SPINE_AUDIT.md, REPRODUCIBILITY_AND_CANONICAL_SUITE.md  
**Purpose:** Single source of truth checklist for ANY agent before running experiments

---

## Quick Reference: Evidence Hierarchy

| Evidence Level | Symbol | Meaning |
|----------------|--------|---------|
| Disk-Verified | ✅ | File exists in `results/` NOW |
| Code-Verified | ⚠️ | Implementation exists in `src/`, no disk artifact |
| Doc-Claimed | ❌ | Claimed in markdown, unverified |

---

## A. Run Artifact Requirements

### Required Files (Every Run Must Produce)

- [ ] `config.json` - Exact config snapshot at runtime
- [ ] `summary.json` - Machine-readable aggregated metrics
- [ ] `metadata.json` - Standardized reproducibility fields (see below)
- [ ] `<experiment>.csv` - Per-sample detailed results
- [ ] `prompt_bank_version.txt` - Hash of bank.json
- [ ] `prompt_bank_version.json` - JSON wrapper: `{"version": "<hash>"}`

### Optional Files

- [ ] `report.md` - Human-readable summary
- [ ] `*.pt` - Steering vectors or activations (if applicable)
- [ ] `logits.pt` - Raw logits (for reproducibility - RECOMMENDED)

### metadata.json Contents (Industry-Grade Contract)

```json
{
  "git_commit": "<40-char hash or 'not_a_git_repo'>",
  "prompt_bank_version": "<16-char SHA256 prefix>",
  "model_id": "mistralai/Mistral-7B-v0.1",
  "seed": 42,
  "n_pairs": 30,
  "eval_window": 16,
  "intervention_scope": "all_tokens | last_16 | BOS_only | first_4",
  "behavior_metric": "mode_score_m",
  "prompt_ids": {
    "recursive": ["L3_deeper_001", "L4_full_002", ...],
    "baseline": ["baseline_math_001", "baseline_factual_002", ...]
  }
}
```

### Run Index (Centralized Ledger)

- [ ] Each run appends to `results/RUN_INDEX.jsonl`
- [ ] Entry includes: timestamp, run_dir, experiment, all metadata fields
- [ ] **KNOWN GAP:** `RUN_INDEX.jsonl` does NOT exist on disk despite code

---

## B. Metric Requirements

### NEW: Baseline Metrics Suite (Nanda-Standard)

**Module:** `src/metrics/baseline_suite.py`

**Required for ALL experiments making causal claims:**

```python
from src.metrics.baseline_suite import BaselineMetricsSuite

suite = BaselineMetricsSuite(model, tokenizer, device)

# Single prompt
metrics = suite.compute_all(prompt)

# Comparison (recursive vs baseline)
comparison = suite.compute_comparison(recursive_prompt, baseline_prompt)

# Batch statistics (for publication)
stats = suite.compute_batch_statistics(recursive_prompts, baseline_prompts)
```

**Metrics Computed:**

| Metric | Type | Purpose | Linear in Residual? |
|--------|------|---------|---------------------|
| `rv` | Geometric | Contraction detection | ❌ No (SVD-based) |
| `logit_diff` | Nanda-standard | Component attribution | ✅ YES |
| `logit_lens` | Analysis | Crystallization point | N/A |
| `mode_score_m` | Behavioral | Mode classification | ❌ No (logsumexp) |
| `activation_norms` | Diagnostic | Intervention effects | ✅ YES |

**Why Both R_V and Logit_diff:**

Per Nanda (2023): *"Logit difference is a fantastic metric because it's a mostly linear function of the residual stream which makes it easy to directly attribute logit difference to individual components."*

- **R_V** (our novel metric) detects geometric contraction but is **nonlinear** → Cannot attribute to specific components
- **logit_diff** (Nanda-standard) is **linear** → Enables proper causal attribution

**Checklist:**
- [ ] Import `BaselineMetricsSuite`
- [ ] Compute `comparison` for each prompt pair
- [ ] Include `stats` in summary.json
- [ ] Report both `rv_delta` AND `logit_diff_delta`

---

### PRIMARY Metric: `mode_score_m` (Logit-Space)

**Definition:**
```
M = logsumexp(logits[R]) - logsumexp(logits[T])
```

**Where:**
- R = Recursive token set (observer, awareness, consciousness, etc.)
- T = Task token set (top-K from baseline logits)

**Implementation:** `src/metrics/mode_score.py::ModeScoreMetric`

**Usage Pattern:**
```python
from src.metrics.mode_score import ModeScoreMetric

metric = ModeScoreMetric(tokenizer, device)
score = metric.compute_score(logits, baseline_logits=baseline_logits)
```

**Required in Summary:**
- [ ] `mode_score_m` - Mean score (baseline or recursive)
- [ ] `mode_score_m_delta` - Change from intervention (if applicable)

### SECONDARY Metric: `rv` (Geometric Signature)

**Definition:**
```
R_V = PR_late / PR_early
```

**Where:**
- PR = Participation Ratio = (Σλᵢ²)² / Σ(λᵢ²)²
- λᵢ = singular values from SVD
- Early layer: 5
- Late layer: num_layers - 5 (typically 27)
- Window: Last 16 tokens

**Implementation:** `src/metrics/rv.py::compute_rv`

**Usage Pattern:**
```python
from src.metrics.rv import compute_rv

rv = compute_rv(model, tokenizer, text, early=5, late=27, window=16, device=device)
```

**Required in Summary:**
- [ ] `rv` - R_V value
- [ ] `rv_baseline_mean`, `rv_ablated_mean`, `rv_delta` (for ablation)
- [ ] `rv_restoration_pct` (for sufficiency tests)

### DEPRECATED Metrics (DO NOT USE for primary claims)

| Metric | Location | Issue | Replacement |
|--------|----------|-------|-------------|
| `recursion_score` | inline in p1_ablation.py, surgical_sweep.py | Regex-based, text-space | `mode_score_m` |
| `behavior_states` | src/metrics/behavior_states.py | Legacy phenomenological | `mode_score_m` |
| `keyword_count` | inline in layer_sweep.py | Non-comparable | `mode_score_m` |

### Restoration Normalization (For Patching Experiments)

**Required for Sufficiency/Patching Tests:**
```
restore_norm(M) = (M_patched - M_corrupt) / (M_clean - M_corrupt)
```

**KNOWN GAP:** `restore_norm` is NOT implemented repo-wide

---

## C. Prompt Handling Requirements

### Canonical Source

- **File:** `prompts/bank.json` (754 prompts)
- **Loader:** `prompts/loader.py::PromptLoader`

### Required Usage Pattern

```python
from prompts.loader import PromptLoader

loader = PromptLoader()

# Get pairs WITH IDs (REQUIRED for reproducibility)
pairs_with_ids = loader.get_balanced_pairs_with_ids(
    n_pairs=30,
    seed=42  # Canonical seed
)

# Returns: List[(rec_id, base_id, rec_text, base_text)]
```

### Version Hash Mechanism

**Definition:** SHA256 hash of `bank.json` file, first 16 characters

**Implementation:**
```python
@property
def version(self) -> str:
    import hashlib
    with open(self.bank_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]
```

### Store in Results

- [ ] `prompt_bank_version.txt` - Plain text hash
- [ ] `prompt_bank_version.json` - JSON wrapper
- [ ] `metadata.json` with `prompt_ids.recursive` and `prompt_ids.baseline`
- [ ] CSV with `recursive_prompt_id` and `baseline_prompt_id` columns

### Canonical Groups

| Group | Use Case |
|-------|----------|
| `L3_deeper`, `L4_full`, `L5_refined` | Recursive prompts |
| `baseline_math`, `baseline_factual`, `baseline_creative` | Baseline prompts |

---

## D. Statistical Standards

### Minimum Sample Sizes

| Experiment Type | Minimum N | Notes |
|-----------------|-----------|-------|
| Exploratory | 30 pairs | Initial investigation |
| Publication-Grade | 80 pairs | Gold standard requirement |
| Cross-Architecture | 50 per model | Per GOLD_STANDARD |

### Significance Thresholds

- [ ] p-value threshold: p < 0.01 (with Bonferroni correction for multiple tests)
- [ ] Effect size threshold: |Cohen's d| ≥ 0.5

### Required Reporting

- [ ] Mean ± Standard Deviation
- [ ] t-statistic and p-value
- [ ] Cohen's d (effect size)
- [ ] 95% Confidence Interval (recommended)
- [ ] Verdict (NECESSARY/NOT NECESSARY, SUFFICIENT/NOT SUFFICIENT)

### Verdict Logic (Ablation)

```python
# CORRECT INTERPRETATION:
# delta = rv_ablated - rv_baseline
# If delta > 0.1 and significant: Layer IS NECESSARY (ablation removes contraction)
# If delta < -0.1 and significant: Layer is NOT necessary (ablation increases contraction)
# If delta ≈ 0: Inconclusive
```

---

## E. Generation Parameter Standards

### Required Logging

- [ ] `max_new_tokens` (default: 200)
- [ ] `temperature` (default: 0.0 for deterministic, 0.7 for sampling)
- [ ] `do_sample` (True/False)
- [ ] `top_p` (if sampling)
- [ ] `pad_token_id` (typically `tokenizer.eos_token_id`)

### Standard Configurations

**Deterministic (for metrics):**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.0,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)
```

**Sampling (for behavior analysis):**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)
```

---

## F. Model Standards

### Reference Model

- **Canonical:** `mistralai/Mistral-7B-v0.1` (Base)
- **Instruct models:** Treated as separate phenotype

### Required Settings

- [ ] `model.eval()` - Always in evaluation mode
- [ ] `torch.no_grad()` - Gradients disabled
- [ ] `torch.float16` - Half precision
- [ ] `device_map="auto"` - Automatic device placement

### Layer Indices

| Parameter | Value | Notes |
|-----------|-------|-------|
| Early layer | 5 | Fixed across models |
| Late layer | num_layers - 5 | Typically 27 for 32-layer |
| Window size | 16 | Token window for PR computation |

---

## G. Hook Standards

### Context Manager Pattern (MANDATORY)

```python
# ALWAYS use context managers for hooks
with capture_v_projection(model, layer_idx=27) as storage:
    with torch.no_grad():
        model(**inputs)
v_tensor = storage["v"]
```

### NEVER Do This

```python
# WRONG: Manual hook registration without cleanup
handle = module.register_forward_hook(hook_fn)
model(**inputs)
# If exception occurs, hook remains attached!
```

---

## H. Pre-Flight Checklist

### Before Running ANY New Experiment

- [ ] **Model:** Load with `load_model()`, call `model.eval()`
- [ ] **Seed:** Call `set_seed(42)` or configured seed
- [ ] **Prompts:** Use `PromptLoader.get_balanced_pairs_with_ids()`
- [ ] **Metrics:** Import `ModeScoreMetric` and `compute_rv`
- [ ] **Config:** Verify all params are in config JSON
- [ ] **Hooks:** Use context managers only

### During Experiment

- [ ] **Log:** Every iteration result to list
- [ ] **Catch:** Exceptions with informative messages
- [ ] **Memory:** Call `torch.cuda.empty_cache()` between large runs

### After Experiment

- [ ] **Save CSV:** `df.to_csv(run_dir / "results.csv")`
- [ ] **Save Summary:** `json.dump(summary, run_dir / "summary.json")`
- [ ] **Save Metadata:** `save_metadata(run_dir, metadata)`
- [ ] **Index:** `append_to_run_index(run_dir, summary)`
- [ ] **Verify:** Check that RUN_INDEX.jsonl was actually created

---

## I. Controls for Causal Claims

### Required Controls (Necessity Test)

- [ ] Baseline control (no intervention)
- [ ] Ablation condition (zero target component)

### Required Controls (Sufficiency Test)

- [ ] Baseline control (no patching)
- [ ] Patch condition (source → target)
- [ ] Restoration percentage: `(M_patched - M_baseline) / (M_recursive - M_baseline)`

### Recommended Controls (Robustness)

- [ ] Random direction control (norm-matched random vectors)
- [ ] Orthogonal direction control
- [ ] Wrong-layer control (same intervention, different layer)
- [ ] Parametric sweep (multiple alpha values)

---

## J. Known Infrastructure Gaps

| Issue | Status | Workaround |
|-------|--------|------------|
| RUN_INDEX.jsonl doesn't exist | ❌ Critical | Manually verify `append_to_run_index()` creates file |
| metadata.json not found in runs | ❌ Critical | Call `save_metadata()` explicitly |
| restore_norm not implemented | ❌ Missing | Compute manually in each pipeline |
| generation_params not logged | ⚠️ Partial | Add to metadata dict explicitly |
| model_revision not logged | ⚠️ Missing | Extract from model.config if available |

---

## K. Quick Copy-Paste Templates

### Industry-Grade Summary Template

```python
summary = {
    "experiment": "experiment_name",
    "n_pairs": len(pairs),
    # PRIMARY: Mode Score M
    "mode_score_m": float(df["mode_baseline"].mean()),
    "mode_score_m_delta": float(df["mode_delta"].mean()),
    "mode_t_statistic": mode_stat,
    "mode_pvalue": mode_pvalue,
    "mode_significant": mode_significant,
    "mode_cohens_d": mode_cohens_d,
    # SECONDARY: R_V signature
    "rv": float(df["rv_baseline"].mean()),
    "rv_baseline_mean": float(df["rv_baseline"].mean()),
    "rv_ablated_mean": float(df["rv_ablated"].mean()),
    "rv_delta_mean": float(df["rv_delta"].mean()),
    "rv_t_statistic": rv_stat,
    "rv_pvalue": rv_pvalue,
    "rv_significant": rv_significant,
    "rv_cohens_d": rv_cohens_d,
    # Standardized metadata
    "eval_window": window_size,
    "intervention_scope": "all_tokens",
    "behavior_metric": "mode_score_m",
    # Merge from get_run_metadata()
    **metadata,
}
```

### Metadata Generation Template

```python
from src.utils.run_metadata import get_run_metadata, save_metadata, append_to_run_index

metadata = get_run_metadata(
    cfg,
    prompt_ids=pairs_with_ids,  # From get_balanced_pairs_with_ids()
    eval_window=16,
    intervention_scope="all_tokens",
    behavior_metric="mode_score_m",
)

save_metadata(run_dir, metadata)
append_to_run_index(run_dir, summary)
```

---

**Checklist Version:** 1.0  
**Last Updated:** January 10, 2026  
**Next Review:** After completion of industry-grade pipeline fixes
