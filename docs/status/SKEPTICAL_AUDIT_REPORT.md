# Skeptical Audit Report: Experimental Infrastructure & Methodology

**Date:** January 5, 2025  
**Purpose:** High-leverage questions about run tracking, reproducibility, metric definitions, and standardization

---

## 1. Run Index: Central Registry Status

### Current State: ❌ **NO CENTRAL INDEX**

**Finding:** There is no single place listing all runs with:
- Script name
- Config file path
- Model version (base vs instruct; v0.1 vs v0.2)
- Prompt bank slice used
- Metric outputs saved

**What Exists:**
- `src/pipelines/registry.py` - Maps experiment names to functions (not runs)
- `results/phase1_mechanism/runs/` - Timestamped directories with individual configs
- Each run directory contains `config.json` (snapshot) and `summary.json` (results)

**Gap:** No master index or database tracking:
- Which experiments ran when
- What model versions were used
- What prompt slices were tested
- Cross-run comparisons

**Recommendation:**
Create `results/RUN_INDEX.json` that tracks:
```json
{
  "runs": [
    {
      "timestamp": "20250105_122534",
      "experiment": "combined_mlp_sufficiency_test",
      "config_path": "configs/combined_mlp_sufficiency_l0_l1.json",
      "model": "mistralai/Mistral-7B-v0.1",
      "model_type": "base",
      "model_version": "v0.1",
      "prompt_bank": "prompts/bank.json",
      "prompt_slice": "balanced_pairs",
      "n_pairs": 30,
      "seed": 42,
      "run_dir": "results/phase1_mechanism/runs/20250105_122534_l0_l1_combined_sufficiency",
      "metrics": ["rv_restoration_pct", "mode_delta", "norm_logs"]
    }
  ]
}
```

---

## 2. Replayability: What's Stored?

### Current State: ⚠️ **PARTIAL REPRODUCIBILITY**

**What IS Stored:**
- ✅ Config snapshot (`config.json` in each run directory)
- ✅ Seed values (hardcoded as `42` in most scripts, or from config)
- ✅ Prompt texts (in CSV files: `recursive_text`, `baseline_text`)
- ✅ Model name (in config)

**What is MISSING:**
- ❌ **Prompt IDs** - Prompts are stored as text, but no reference to `prompts/bank.json` indices
- ❌ **Prompt bank version hash** - No tracking of which version of `prompts/bank.json` was used
- ❌ **Exact intervention configs** - Some experiments store intervention params, but not all
- ❌ **Generation parameters** - `temperature`, `do_sample`, `max_new_tokens` sometimes hardcoded

**Example from `mlp_ablation_necessity.py`:**
```python
set_seed(42)  # Hardcoded, not from config
pairs = loader.get_balanced_pairs(n_pairs=n_pairs, seed=42)  # Seed hardcoded
# Prompts stored as text in CSV, but no prompt IDs
```

**Gap:** Cannot reproduce exact forward pass because:
1. Prompt selection is non-deterministic if `prompts/bank.json` changes
2. No prompt bank version tracking
3. Some generation params are hardcoded

**Recommendation:**
1. Store prompt IDs in CSV: `prompt_id`, `recursive_prompt_id`, `baseline_prompt_id`
2. Store prompt bank version hash in `summary.json`:
   ```python
   summary["prompt_bank_version"] = loader.version  # Already exists in PromptLoader
   ```
3. Move all hardcoded seeds to config
4. Store full generation params in config snapshot

---

## 3. Behavior Target Definition: "Recursive Mode" in Logit Space

### Current Definition: ✅ **CLEARLY DEFINED**

**Location:** `src/metrics/mode_score.py`

**Definition:**
```python
M = logsumexp(logits[R]) - logsumexp(logits[T])
```

Where:
- **R (Recursive tokens):** Fixed token set (30 tokens) built from keywords:
  - `observer`, `observed`, `awareness`, `itself`, `self`, `recognition`, `consciousness`, `witness`, `reflection`, `recursive`, `loop`, `meta`, etc.
  - Includes variations (case, spacing, plurals, compounds)
  - Token IDs extracted via tokenizer

- **T (Task tokens):** Dynamic set from baseline logits
  - Top-K per position (default K=10)
  - OR union of top-K across all positions
  - Defined relative to baseline prompt

**Type:** **Logit difference** (not a classifier, not prompt-conditional)

**Key Properties:**
- M > 0: Recursive tokens have higher log-probability than task tokens
- M < 0: Task tokens dominate
- Mean over sequence: `m.mean().item()`

**Code Reference:**
```python
# src/metrics/mode_score.py, lines 161-224
def compute_score(self, logits, baseline_logits=None, top_k_task=10, per_position=True):
    # Extract recursive token logits
    r_logits = logits[:, self.recursive_token_ids]  # (N, n_recursive_tokens)
    lse_r = torch.logsumexp(r_logits, dim=-1)  # (N,)
    
    # Extract task token logits from baseline
    _, task_indices = torch.topk(baseline_logits, k=top_k_task, dim=-1)
    t_logits = torch.gather(logits, dim=-1, index=task_indices)
    lse_t = torch.logsumexp(t_logits, dim=-1)  # (N,)
    
    m = lse_r - lse_t  # Mode Score
    return m.mean().item()
```

**Validation:** ✅ Token set is validated at initialization (prints token count and range)

---

## 4. Common Eval Harness: Central Metrics?

### Current State: ⚠️ **PARTIALLY CENTRALIZED**

**What Exists:**
- ✅ `src/metrics/rv.py` - R_V computation (centralized)
- ✅ `src/metrics/mode_score.py` - Mode Score M (centralized)
- ✅ `src/metrics/behavior_strict.py` - StrictBehaviorScore (centralized)

**What's Embedded Per-Script:**
- ❌ Some scripts compute metrics inline (e.g., `participation_ratio` in `ioi_causal_test.py`)
- ❌ No `evaluate_run.py` - Each script computes metrics independently
- ❌ No `metrics.py` aggregator - Metrics are imported but not standardized

**Example Inconsistency:**
- `mlp_steering_sweep.py` uses `compute_rv()` from `src/metrics/rv.py` ✅
- `ioi_causal_test.py` defines `participation_ratio()` inline ❌
- `geometry_behavior.py` defines `participation_ratio()` inline ❌

**Gap:** No common evaluation harness that:
- Loads a run directory
- Recomputes all metrics from stored activations/logits
- Validates consistency

**Recommendation:**
Create `src/eval/evaluate_run.py`:
```python
def evaluate_run(run_dir: Path, recompute: bool = False):
    """Load run results and recompute/validate metrics."""
    config = load_config(run_dir / "config.json")
    results = load_csv(run_dir / "results.csv")
    
    # Recompute metrics if requested
    if recompute:
        metrics = compute_all_metrics(results, config)
        validate_against_stored(results, metrics)
    
    return metrics
```

---

## 5. KV Tests: Behavior Metric Definition

### Finding: ⚠️ **METRIC NOT CLEARLY DOCUMENTED**

**KV Patching Experiments:**
- `src/pipelines/kv_mechanism.py` - Tests geometry transfer (R_V)
- `src/pipelines/kv_sufficiency_matrix.py` - Tests behavior transfer

**Claim:** "~80% behavior transfer" (from audit document)

**Investigation:**
- `kv_mechanism.py` measures **R_V** (geometry), not behavior
- `kv_sufficiency_matrix.py` uses **`label_behavior_state()`** from `src/metrics/behavior_states.py`

**Behavior Metric Definition:**
```python
# src/pipelines/kv_sufficiency_matrix.py:35-37
def _is_expression(text: str) -> bool:
    s = label_behavior_state(text).state
    return s in (BehaviorState.RECURSIVE_PROSE, BehaviorState.NAKED_LOOP)
```

**Metric Type:** **Text classifier** (not logit-based)

**States:**
- `BehaviorState.RECURSIVE_PROSE` - Recursive prose detected
- `BehaviorState.NAKED_LOOP` - Explicit recursive loop detected
- Other states: `BASELINE`, `COLLAPSED`, etc.

**Gap:** This is a **different metric** than Mode Score M:
- **Mode Score M:** Logit-space metric (logsumexp difference) - `src/metrics/mode_score.py`
- **Behavior State:** Text-space classifier (pattern matching) - `src/metrics/behavior_states.py`

**Behavior State Classifier:**
- Heuristic text classifier (not logit-based)
- Detects: `RECURSIVE_PROSE`, `NAKED_LOOP`, `BASELINE`, `QUESTIONING`, `COLLAPSE`
- Based on: keyword matching, repetition ratio, question marks, identity equations

**The "80% behavior transfer" claim:**
- **Metric:** `label_behavior_state()` → `is_expression` (boolean: RECURSIVE_PROSE or NAKED_LOOP)
- **Not comparable to Mode Score M** (different measurement space)
- **Needs clarification:** What was the baseline? What was the transfer rate?

**Recommendation:**
1. Document which metric was used in each experiment
2. Add to run summary: `"behavior_metric": "behavior_state"` or `"mode_score_m"`
3. Do NOT compare "80% behavior transfer" (text classifier) with Mode Score M (logit metric)

**Recommendation:**
1. Check `kv_sufficiency_matrix.py` for behavior metric definition
2. Document in code comments: "Behavior metric: Mode Score M (see `src/metrics/mode_score.py`)"
3. Add to run summary: `"behavior_metric": "mode_score_m"`

---

## 6. Window Mismatch: Standardization Issue

### Current State: ❌ **INCONSISTENT**

**The Problem:**
- **R_V measurement:** Always uses last 16 tokens (window=16)
- **Interventions:** Vary in scope:
  - **Last-16 patching:** `circuit_discovery.py`, `p1_ablation.py` patch last 16 tokens
  - **All-position ablation:** `mlp_ablation_necessity.py` ablates ALL tokens
  - **Position-specific:** `mlp_ablation_position_specific.py` tests BOS, first-4, last-16, all

**Evidence:**
```python
# mlp_ablation_necessity.py - Ablates ALL positions
class MLPAblationHook:
    def hook_fn(module, inp, out):
        return torch.zeros_like(out)  # ALL tokens zeroed

# circuit_discovery.py - Patches last W tokens
out_p[:, -W:, :] = source[:, -W:, :]  # Only last W tokens patched
```

**Impact:**
- L0 ablation affects ALL tokens, but R_V is measured on last 16
- This is actually CORRECT (we want to test if early-layer ablation affects late-layer geometry)
- But it's confusing: "Where we intervene" ≠ "Where we measure"

**Recommendation:**
**Standardize "where we measure" (last-16) while allowing interventions to vary:**

1. **Document the distinction:**
   ```python
   # Standard measurement window (always last 16 tokens)
   MEASUREMENT_WINDOW = 16
   
   # Intervention scope (varies by experiment)
   INTERVENTION_SCOPE = "all_tokens"  # or "last_16", "first_4", "BOS_only"
   ```

2. **Add to config:**
   ```json
   {
     "params": {
       "measurement_window": 16,  // Always 16 for R_V
       "intervention_scope": "all_tokens"  // Varies
     }
   }
   ```

3. **Clarify in docstrings:**
   ```python
   """
   Ablates L0 MLP at ALL token positions.
   
   Note: R_V is measured on last 16 tokens (standard window).
   This tests if early-layer ablation affects late-layer geometry.
   """
   ```

**Verdict:** The mismatch is intentional and correct, but needs documentation.

---

## Summary Table

| Question | Status | Action Required |
|----------|--------|----------------|
| **Run Index** | ❌ Missing | Create `results/RUN_INDEX.json` |
| **Replayability** | ⚠️ Partial | Store prompt IDs, bank version, all params |
| **Behavior Target** | ✅ Defined | Document in README |
| **Common Eval Harness** | ⚠️ Partial | Create `evaluate_run.py` |
| **KV Behavior Metric** | ⚠️ Unclear | Document in `kv_sufficiency_matrix.py` |
| **Window Mismatch** | ⚠️ Inconsistent | Document distinction, standardize config |

---

## Priority Actions

1. **HIGH:** Document "recursive mode" definition in README
2. **HIGH:** Standardize window documentation (measurement vs intervention)
3. **MEDIUM:** Create run index tracker
4. **MEDIUM:** Store prompt IDs and bank version in results
5. **LOW:** Create common eval harness

---

## Code References

- **Mode Score Definition:** `src/metrics/mode_score.py:161-224`
- **R_V Computation:** `src/metrics/rv.py:compute_rv()`
- **Experiment Registry:** `src/pipelines/registry.py`
- **Run Directory Creation:** `src/core/experiment_io.py:create_run_dir()`
- **Window Usage:** `grep -r "window.*16" src/pipelines/`

---

**Next Steps:**
1. Review `kv_sufficiency_matrix.py` for behavior metric
2. Create run index tracker script
3. Update all experiments to store prompt IDs
4. Document window distinction in `.cursorrules`
