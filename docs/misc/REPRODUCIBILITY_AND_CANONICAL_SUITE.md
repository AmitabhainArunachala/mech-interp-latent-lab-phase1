# Reproducibility Audit & Canonical Experiment Suite

**Date:** January 5, 2025  
**Purpose:** Answer critical reproducibility questions and define canonical experiment suite

---

## 1. Logits/Activations Storage: Do Older Runs Store Raw Data?

### Finding: ❌ **NO LOGITS STORED**

**Search Results:**
- ✅ Found `.pt` files: **Steering vectors only** (not logits/activations)
  - `results/runs/20251217_135538_steering_layer_matrix/steering_vector_L*.pt`
  - `results/runs/20251217_130651_steering_analysis/steering_vectors/steering_vector_*.pt`
- ❌ **No `.npy` files** (numpy arrays)
- ❌ **No `.npz` files** (compressed numpy arrays)
- ❌ **No `logits.npy` or `activations.npz` files**

**What IS Stored:**
- CSV files with metrics (R_V, mode score, coherence)
- JSON summaries (aggregated statistics)
- Generated text (in CSV)
- Steering vectors (`.pt` files) - but only for steering experiments

**Verdict:** **We need to re-run experiments** to get logits/activations for:
- Exact forward pass reproduction
- Recomputing metrics with different methods
- Cross-experiment comparisons

**Recommendation:**
Add logits/activations storage to future experiments:
```python
# Save logits for reproducibility
torch.save(logits, run_dir / "logits.pt")
# Or compressed numpy
np.savez_compressed(run_dir / "activations.npz", 
                    logits=logits.cpu().numpy(),
                    hidden_states=hidden_states.cpu().numpy())
```

---

## 2. PromptLoader Version Hash Mechanism

### Finding: ✅ **CLEARLY DEFINED**

**Location:** `prompts/loader.py:110-120`

**Implementation:**
```python
@property
def version(self) -> str:
    """
    Get prompt bank version (hash of bank.json).
    
    Returns:
        Short hash string for version tracking.
    """
    import hashlib
    with open(self.bank_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]
```

**Mechanism:**
- SHA256 hash of entire `prompts/bank.json` file
- First 16 characters of hex digest
- Example: `"a3f2b1c4d5e6f7a8"`

**Usage in Scripts:**
- ✅ **Newer scripts (2024-12-15+) store version:**
  - `circuit_discovery.py` (line 45-47)
  - `mlp_ablation_necessity.py` (line 100-102)
  - `mlp_steering_sweep.py` (line 188-190)
  - `mlp_sufficiency_test.py` (line 115-117)
  - `mlp_combined_sufficiency_test.py` (line 165-167)
  - `behavior_strict.py` (line 151-153, 323)
  - `kv_sufficiency_matrix.py` (line 292-295)
  - And 20+ more scripts

**Storage Format:**
```python
bank_version = loader.version
(run_dir / "prompt_bank_version.txt").write_text(bank_version)
(run_dir / "prompt_bank_version.json").write_text(
    json.dumps({"version": bank_version}, indent=2) + "\n"
)
# Also in summary.json:
summary["prompt_bank_version"] = bank_version
```

**Verdict:** ✅ Version tracking is **implemented and widely used** in newer scripts.

---

## 3. "balanced_pairs" Definition

### Finding: ✅ **CLEARLY DEFINED**

**Location:** `prompts/loader.py:147-194`

**Definition:**
```python
def get_balanced_pairs(
    self,
    n_pairs: int = 30,
    recursive_groups: Optional[List[str]] = None,
    baseline_groups: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Generate balanced recursive/baseline prompt pairs.
    
    Args:
        n_pairs: Number of pairs to generate.
        recursive_groups: List of recursive groups to sample from.
                         Default: ["L3_deeper", "L4_full", "L5_refined"].
        baseline_groups: List of baseline groups to sample from.
                         Default: ["baseline_math", "baseline_factual", "baseline_creative"].
        seed: Random seed.
    
    Returns:
        List of (recursive_prompt, baseline_prompt) tuples.
    """
    rng = random.Random(seed)
    
    if recursive_groups is None:
        recursive_groups = ["L3_deeper", "L4_full", "L5_refined"]
    if baseline_groups is None:
        baseline_groups = ["baseline_math", "baseline_factual", "baseline_creative"]
    
    # Filter prompts by group
    recursive = []
    baseline = []
    for k, v in self.prompts.items():
        if v.get("group") in recursive_groups:
            recursive.append(v["text"])
        elif v.get("group") in baseline_groups:
            baseline.append(v["text"])
    
    # Sample with seed
    n_rec = min(n_pairs, len(recursive))
    n_base = min(n_pairs, len(baseline))
    sampled_rec = rng.sample(recursive, n_rec)
    sampled_base = rng.sample(baseline, n_base)
    
    # Pair them up
    pairs = []
    for i in range(min(n_rec, n_base)):
        pairs.append((sampled_rec[i], sampled_base[i]))
    
    return pairs
```

**Key Properties:**
- ✅ **Deterministic with seed:** Same seed → same pairs
- ⚠️ **NOT deterministic across bank changes:** If `prompts/bank.json` changes, different prompts may be selected
- ⚠️ **No prompt IDs stored:** Returns text tuples, not prompt keys

**Problem:** If `prompts/bank.json` changes, `get_balanced_pairs(seed=42)` may return different prompts.

**Solution Needed:**
Store prompt IDs (keys from `bank.json`) in results:
```python
# Instead of just text:
pairs = loader.get_balanced_pairs(n_pairs=30, seed=42)

# Store prompt keys:
pairs_with_ids = []
for rec_text, base_text in pairs:
    rec_id = loader.find_prompt_id(rec_text)  # Need to add this method
    base_id = loader.find_prompt_id(base_text)
    pairs_with_ids.append((rec_id, base_id, rec_text, base_text))
```

---

## 4. Deterministic Prompt ID Selection

### Current State: ⚠️ **PARTIALLY DETERMINISTIC**

**What Works:**
- ✅ Same seed + same `bank.json` → same prompts
- ✅ Version hash tracks `bank.json` changes

**What Doesn't Work:**
- ❌ If `bank.json` changes, same seed may select different prompts
- ❌ No prompt IDs stored in results (only text)

**Recommendation:**
1. **Add `get_balanced_pairs_with_ids()` method:**
```python
def get_balanced_pairs_with_ids(
    self,
    n_pairs: int = 30,
    recursive_groups: Optional[List[str]] = None,
    baseline_groups: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Tuple[str, str, str, str]]:  # (rec_id, base_id, rec_text, base_text)
    """Same as get_balanced_pairs but returns prompt IDs."""
    pairs = self.get_balanced_pairs(n_pairs, recursive_groups, baseline_groups, seed)
    pairs_with_ids = []
    for rec_text, base_text in pairs:
        rec_id = self._find_prompt_id(rec_text)
        base_id = self._find_prompt_id(base_text)
        pairs_with_ids.append((rec_id, base_id, rec_text, base_text))
    return pairs_with_ids

def _find_prompt_id(self, text: str) -> Optional[str]:
    """Find prompt ID by text (exact match)."""
    for k, v in self.prompts.items():
        if v["text"] == text:
            return k
    return None
```

2. **Store prompt IDs in CSV:**
```python
results.append({
    "recursive_prompt_id": rec_id,
    "baseline_prompt_id": base_id,
    "recursive_text": rec_text,
    "baseline_text": base_text,
    ...
})
```

3. **Store prompt ID list in summary:**
```python
summary["prompt_ids"] = {
    "recursive": [rec_id for rec_id, _, _, _ in pairs_with_ids],
    "baseline": [base_id for _, base_id, _, _ in pairs_with_ids]
}
```

---

## 5. Canonical Experiment Suite (10-15 Key Experiments)

Based on `MISTRAL_7B_COMPLETE_CAUSAL_MAP.md` and causal story arc, here are the **essential experiments** that define the repo's findings:

### Phase 0: Metric Validation

#### 1. **phase0_metric_targets**
- **Script:** `src/pipelines/phase0_metric_targets.py`
- **Config:** `configs/phase0_metric_targets.json`
- **Purpose:** Validate R_V metric computation (PR at different layers)
- **Key Finding:** R_V measured correctly at L5/L27
- **Status:** ✅ Foundation

#### 2. **phase0_minimal_pairs**
- **Script:** `src/pipelines/phase0_minimal_pairs.py`
- **Config:** `configs/phase0_minimal_pairs.json`
- **Purpose:** Establish baseline R_V separation (recursive vs baseline)
- **Key Finding:** R_V < 1.0 for recursive, R_V ≈ 1.0 for baseline
- **Status:** ✅ Foundation

---

### Phase 1: Causal Discovery

#### 3. **circuit_discovery** ⭐ **CRITICAL**
- **Script:** `src/pipelines/circuit_discovery.py`
- **Config:** `configs/gold/11_circuit_discovery.json`
- **Purpose:** Attribution patching sweep (identify causal drivers)
- **Key Finding:** L0 MLP attribution = 1.67 (highest), L18-L20 MLPs also strong
- **Evidence:** `CIRCUIT_DISCOVERY_REPORT.md`
- **Status:** ✅ **MUST RE-RUN** (found L0 MLP)

#### 4. **mlp_ablation_necessity** ⭐ **CRITICAL**
- **Script:** `src/pipelines/mlp_ablation_necessity.py`
- **Config:** `configs/mlp_ablation_necessity_l0.json` (and L1, L2, L3)
- **Purpose:** Test if L0-L3 MLPs are NECESSARY (zero ablation)
- **Key Finding:** L0 ablation → R_V delta = +0.76 (contraction disappears), p < 10⁻²⁵
- **Evidence:** `results/phase1_mechanism/runs/20260105_103250_l0_necessity_test/`
- **Status:** ✅ **MUST RE-RUN** (proves necessity)

#### 5. **mlp_sufficiency_test** ⭐ **CRITICAL**
- **Script:** `src/pipelines/mlp_sufficiency_test.py`
- **Config:** `configs/mlp_sufficiency_l0.json`
- **Purpose:** Test if L0 MLP alone is SUFFICIENT (patch recursive → baseline)
- **Key Finding:** L0 alone NOT sufficient (R_V restoration = -68.4%)
- **Status:** ✅ **MUST RE-RUN** (proves insufficiency)

#### 6. **mlp_combined_sufficiency_test** ⭐ **CRITICAL**
- **Script:** `src/pipelines/mlp_combined_sufficiency_test.py`
- **Config:** `configs/combined_mlp_sufficiency_l0_l1.json`
- **Purpose:** Test if L0+L1 together are SUFFICIENT
- **Key Finding:** (Running now - Jan 5, 2025)
- **Status:** 🔄 **IN PROGRESS**

#### 7. **position_specific_ablation** ⭐ **CRITICAL**
- **Script:** `src/pipelines/mlp_ablation_position_specific.py`
- **Config:** `configs/position_specific_l0_ablation.json`
- **Purpose:** Test which token positions drive L0 effect (BOS, first-4, last-16, all)
- **Key Finding:** Position-distributed effect (BOS + last-16 both significant)
- **Status:** ✅ **MUST RE-RUN** (tests position specificity)

---

### Phase 1B: Transfer & Steering

#### 8. **mlp_steering_sweep** ⭐ **CRITICAL**
- **Script:** `src/pipelines/mlp_steering_sweep.py`
- **Config:** `configs/mlp_steering_sweep_corrected.json`
- **Purpose:** Test MLP steering at all layers (find optimal transfer layers)
- **Key Finding:** L3-L4 optimal for steering (not L0), L2 is artifact
- **Evidence:** `MLP_STEERING_STATUS_REPORT.md`
- **Status:** ✅ **MUST RE-RUN** (finds transferability)

#### 9. **random_direction_control** ⭐ **CRITICAL**
- **Script:** `src/pipelines/random_direction_control.py`
- **Config:** `configs/random_direction_control_l3_targeted.json`
- **Purpose:** Test if steering effects are direction-specific (not artifacts)
- **Key Finding:** L2 steering = artifact (random vectors show similar effects)
- **Status:** ✅ **MUST RE-RUN** (validates steering)

---

### Phase 1C: Late-Layer Attention

#### 10. **p1_ablation** ⭐ **CRITICAL**
- **Script:** `src/pipelines/p1_ablation.py`
- **Config:** `configs/gold/p1_ablation.json` (or similar)
- **Purpose:** Test component hierarchy (V-Proj, Residual, KV cache)
- **Key Finding:** V-Proj primary, Residual amplifier, KV necessary but not sufficient
- **Evidence:** `P1_ABLATION_ANALYSIS.md`
- **Status:** ✅ **MUST RE-RUN** (proves late-layer roles)

#### 11. **surgical_sweep** ⭐ **CRITICAL**
- **Script:** `src/pipelines/surgical_sweep.py`
- **Config:** `configs/gold/15_surgical_sweep.json` (C2 config)
- **Purpose:** Optimal steering configuration (H18+H26 + Residual + KV)
- **Key Finding:** C2 config → 0.15 recursion score, 20% success rate
- **Evidence:** `SURGICAL_SWEEP_DEEP_ANALYSIS.md`
- **Status:** ✅ **MUST RE-RUN** (optimal config)

---

### Phase 1D: KV Cache Mechanism

#### 12. **kv_mechanism** ⭐ **CRITICAL**
- **Script:** `src/pipelines/kv_mechanism.py`
- **Config:** `configs/kv_mechanism.json` (or similar)
- **Purpose:** Test KV cache geometry transfer
- **Key Finding:** KV replacement → 94% geometry transfer
- **Evidence:** `FINAL_REPORT_DEC19.md`
- **Status:** ✅ **MUST RE-RUN** (proves KV role)

#### 13. **kv_sufficiency_matrix**
- **Script:** `src/pipelines/kv_sufficiency_matrix.py`
- **Config:** `configs/kv_sufficiency_matrix.json` (or similar)
- **Purpose:** Test KV cache behavior transfer (with controls)
- **Key Finding:** (Need to check results)
- **Status:** ⚠️ **CHECK IF NEEDED**

---

### Phase 1E: Verification

#### 14. **verification_sweep**
- **Script:** `src/pipelines/verification_sweep.py`
- **Config:** `configs/verification_sweep.json` (or similar)
- **Purpose:** Comprehensive verification with controls
- **Key Finding:** (Need to check results)
- **Status:** ⚠️ **CHECK IF NEEDED**

---

## Summary: Canonical Suite (13 Experiments)

| # | Experiment | Script | Config | Status | Priority |
|---|------------|--------|--------|--------|----------|
| 1 | phase0_metric_targets | `phase0_metric_targets.py` | `configs/phase0_metric_targets.json` | ✅ Foundation | HIGH |
| 2 | phase0_minimal_pairs | `phase0_minimal_pairs.py` | `configs/phase0_minimal_pairs.json` | ✅ Foundation | HIGH |
| 3 | **circuit_discovery** | `circuit_discovery.py` | `configs/gold/11_circuit_discovery.json` | ⭐ **MUST RE-RUN** | **CRITICAL** |
| 4 | **mlp_ablation_necessity** | `mlp_ablation_necessity.py` | `configs/mlp_ablation_necessity_l0.json` | ⭐ **MUST RE-RUN** | **CRITICAL** |
| 5 | **mlp_sufficiency_test** | `mlp_sufficiency_test.py` | `configs/mlp_sufficiency_l0.json` | ⭐ **MUST RE-RUN** | **CRITICAL** |
| 6 | **mlp_combined_sufficiency** | `mlp_combined_sufficiency_test.py` | `configs/combined_mlp_sufficiency_l0_l1.json` | 🔄 **IN PROGRESS** | **CRITICAL** |
| 7 | **position_specific_ablation** | `mlp_ablation_position_specific.py` | `configs/position_specific_l0_ablation.json` | ⭐ **MUST RE-RUN** | **CRITICAL** |
| 8 | **mlp_steering_sweep** | `mlp_steering_sweep.py` | `configs/mlp_steering_sweep_corrected.json` | ⭐ **MUST RE-RUN** | **CRITICAL** |
| 9 | **random_direction_control** | `random_direction_control.py` | `configs/random_direction_control_l3_targeted.json` | ⭐ **MUST RE-RUN** | **CRITICAL** |
| 10 | **p1_ablation** | `p1_ablation.py` | `configs/gold/p1_ablation.json` | ⭐ **MUST RE-RUN** | **CRITICAL** |
| 11 | **surgical_sweep** | `surgical_sweep.py` | `configs/gold/15_surgical_sweep.json` | ⭐ **MUST RE-RUN** | **CRITICAL** |
| 12 | **kv_mechanism** | `kv_mechanism.py` | `configs/kv_mechanism.json` | ⭐ **MUST RE-RUN** | **CRITICAL** |
| 13 | kv_sufficiency_matrix | `kv_sufficiency_matrix.py` | `configs/kv_sufficiency_matrix.json` | ⚠️ Check | MEDIUM |

**Total: 13 experiments** (10-15 range as requested)

---

## Re-Run Requirements

### Why Re-Run?
1. ❌ **No logits stored** - Cannot reproduce exact forward passes
2. ⚠️ **Prompt IDs not stored** - Cannot guarantee same prompts if bank changes
3. ✅ **Version hashes stored** - Can verify prompt bank version
4. ✅ **Configs stored** - Can reproduce experimental setup

### What to Add to Re-Runs:
1. **Store logits:** `torch.save(logits, run_dir / "logits.pt")`
2. **Store prompt IDs:** Add `recursive_prompt_id`, `baseline_prompt_id` to CSV
3. **Store prompt ID list:** Add to `summary.json`
4. **Store activations (optional):** For key layers (L0, L1, L27)

---

## Next Steps

1. **Create re-run script:** `scripts/rerun_canonical_suite.py`
2. **Update PromptLoader:** Add `get_balanced_pairs_with_ids()`
3. **Update all canonical experiments:** Store prompt IDs + logits
4. **Run canonical suite:** Sequential or parallel execution
5. **Validate reproducibility:** Compare new results with old (where possible)

---

**Last Updated:** January 5, 2025

