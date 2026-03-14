# CODE REPRODUCIBILITY AUDIT

**Auditor**: Claude Opus 4.6 (automated)
**Date**: 2026-03-08
**Scope**: All code files used to produce R_V metric results
**Standard**: NeurIPS reproducibility checklist

---

## 1. PR Implementation Comparison (Line-by-Line Diff)

### Two Implementations Found

| Property | `src/metrics/rv.py` (Canonical) | `geometric_lens/metrics.py` (GeometricLens) |
|---|---|---|
| Function signature | `participation_ratio(v_tensor, window_size=16)` | `participation_ratio(tensor, window_size=16)` |
| None guard | Returns `float("nan")` | Returns `float("nan")` |
| Batch handling | `v_tensor[0]` if dim==3 | `tensor[0]` if dim==3 |
| Short-sequence guard | `if T < window_size: return NaN` + logs warning | `if T < window_size: return NaN` (no warning) |
| **SVD precision** | `.double()` on GPU tensor | `.cpu().double()` (forces CPU first) |
| **NaN/Inf guard** | **NONE** | `torch.isnan(v_cpu).any() or torch.isinf(v_cpu).any()` |
| SVD target | `v_window.T` (D x W matrix) | `v_cpu.T` (D x W matrix) |
| SVD full_matrices | `False` | `False` |
| S conversion | `S.cpu().numpy()` | `S.numpy()` (already CPU) |
| Degeneracy check | `total_variance < 1e-10` | `total_variance < 1e-10` |
| **Dead code** | Line 86: `p = S_sq / total_variance` (computed but never used) | None |
| PR formula | `(S_sq.sum()**2) / (S_sq**2).sum()` | `(S_sq.sum()**2) / (S_sq**2).sum()` |
| Error handling | `except Exception: return NaN` | `except Exception: return NaN` |

### Critical Differences

**DIFFERENCE 1 (MEDIUM SEVERITY): SVD computation device**
- `src/metrics/rv.py` computes SVD on whatever device the tensor is on (GPU by default), then moves to CPU for numpy.
- `geometric_lens/metrics.py` explicitly moves to CPU before SVD via `.cpu().double()`.
- **Impact**: On GPU, cusolver can produce slightly different singular values than CPU LAPACK. More critically, cusolver can fail with "CUDA device-side assert" on certain inputs, which the geometric_lens version explicitly avoids (as its comment states). The canonical version is fragile on GPU.
- **Verdict**: geometric_lens is more robust. This difference could cause numerical disagreements in edge cases, though the PR formula is stable enough that typical values should match to ~6 decimal places.

**DIFFERENCE 2 (LOW-MEDIUM SEVERITY): NaN/Inf guard**
- `geometric_lens/metrics.py` checks for NaN/Inf in activations before SVD.
- `src/metrics/rv.py` does not.
- **Impact**: fp16 overflow at deep layers (e.g., layer 30+ on some models) can produce Inf values. The canonical version will crash or produce garbage; geometric_lens returns NaN gracefully.

**DIFFERENCE 3 (COSMETIC): Dead code**
- `src/metrics/rv.py` line 86: `p = S_sq / total_variance` is computed but never used.

### Verdict: Are they the same?
The core formula is identical: `PR = (sum(sigma_i^2))^2 / sum(sigma_i^4)`. The SVD is applied to the transposed window matrix in both cases. For well-behaved inputs (no NaN, no Inf, GPU SVD converges), they produce identical results. For edge cases, geometric_lens is strictly more robust.

**RISK FOR PAPER**: LOW. Results produced by either implementation agree on standard inputs. But the existence of two implementations is itself a reproducibility concern -- a reviewer could run one and get slightly different numbers than the other.

---

## 2. Bug Inventory

### BUG-01: Qwen2.5-7B Layer Count (MEDIUM SEVERITY)

**File**: `geometric_lens/models.py`, line 218-219
**Issue**: Registry claims `num_layers=32` for `Qwen/Qwen2.5-7B`.
**Actual**: Qwen2.5-7B has 28 hidden layers (`num_hidden_layers=28` in HF config).
**Impact**: `late_layer=27` would be the LAST layer (index 27 of 28), not 84% depth. This means the auto-detect `late_layer` calculation is wrong for Qwen. If someone uses the registry spec directly, they get near-final-layer measurements that may behave differently from the intended ~84% depth point.
**Mitigation**: The `auto_detect()` method falls back to reading `model.config.num_hidden_layers` at runtime, so if the model is actually loaded and auto-detected (rather than using the hardcoded spec), the correct value is used. But the hardcoded spec is misleading and will be used when the model_name matches the registry key.

**UPDATE (2026-03-08 verification)**: I checked the HuggingFace model card for `Qwen/Qwen2.5-7B`. The model config shows `"num_hidden_layers": 28`. The registry entry claims 32, sets `late_layer=27`. If 28 layers, then `late_layer` should be ~23 (28 * 0.84 = 23.5). The current setting of 27 would be layer index 27 out of 0-27, i.e., the absolute last layer. This is wrong.

### BUG-02: Validated Patching Script Uses Wrong Model (LOW-MEDIUM SEVERITY)

**File**: `archive/rv_paper_code/VALIDATED_mistral7b_layer27_activation_patching.py`, line 87-88
**Issue**: The docstring and usage example reference `Mistral-7B-Instruct-v0.2`, but the canonical pipeline (`src/core/models.py`) defaults to `Mistral-7B-v0.1` (base model). The code comment says "Instruct models are treated as a separate phenotype (confounding factor)."
**Impact**: If someone follows the validated script's instructions, they load the Instruct model. If they use the canonical pipeline, they load the Base model. These are different models with potentially different R_V behavior.
**Note**: The R_V_PAPER results were originally produced on v0.2 Instruct per the script, but the canonical pipeline defaults to v0.1 Base. This discrepancy must be resolved before submission.

### BUG-03: Dead Code in `src/metrics/rv.py` (COSMETIC)

**File**: `src/metrics/rv.py`, line 86
**Code**: `p = S_sq / total_variance  # Normalized eigenvalues`
**Issue**: Variable `p` is computed but never used. Looks like a leftover from an earlier version that computed effective rank alongside PR.

### BUG-04: Validated Patching Script PR Uses Different Matrix Orientation (MEDIUM SEVERITY)

**File**: `archive/rv_paper_code/VALIDATED_mistral7b_layer27_activation_patching.py`, line 193
**Issue**: `torch.linalg.svd(V_window.float(), full_matrices=False)` -- operates on `V_window` directly (W x D matrix), NOT `V_window.T` (D x W matrix) as in both canonical implementations.
**Impact**: SVD of M vs SVD of M^T produces the same singular values, so PR is identical. BUT the validated script also does NOT cast to float64 (uses `.float()` = float32). This means:
  - The 2x2 validated script uses float32 SVD on GPU.
  - The canonical pipeline uses float64 SVD.
  - The geometric_lens uses float64 SVD on CPU.
  - **Numerical disagreements are possible** for edge cases.

Additionally, the validated script's PR also allows short sequences (no minimum window check -- line 188 allows `V_window.shape[0] < 2` but only checks at line 189), whereas canonical requires `T >= window_size`.

### BUG-05: `compute_rv_with_components` API Mismatch (MEDIUM SEVERITY)

**File**: `src/metrics/rv.py` vs `geometric_lens/metrics.py`
**Issue**: These have DIFFERENT function signatures:
  - `src/metrics/rv.py::compute_rv_with_components(model, tokenizer, text, early, late, window, device)` -- takes model+tokenizer+text, does the full pipeline internally (tokenize, hook, forward pass, PR).
  - `geometric_lens/metrics.py::compute_rv_with_components(v_early, v_late, window)` -- takes pre-captured tensors.
**Impact**: The canonical pipelines ALL import from `src.metrics.rv`, never from `geometric_lens.metrics` for the full R_V computation. The `geometric_lens` version requires the caller to manage hooks. This is a design difference, not a bug per se, but means the two modules are NOT drop-in replacements despite having the same function names.

### BUG-06: Double Forward Pass in `src/metrics/rv.py` (PERFORMANCE/CLARITY)

**File**: `src/metrics/rv.py`, lines 146-154
**Issue**: `compute_rv_with_components()` runs the model TWICE (once per layer capture), because each `capture_v_projection` context manager runs its own forward pass. The `geometric_lens` probe does the same thing (lines 222-230 of `probe.py`). This is wasteful but not a correctness issue. For a NeurIPS reproducibility appendix, a reviewer might question why two forward passes are needed when `capture_multi_layer` in `geometric_lens/hooks.py` can capture both in one pass.

### BUG-07: Duplicate Import in Registry (COSMETIC)

**File**: `src/pipelines/registry.py`, lines 128-133
**Issue**: `run_rv_l27_kv_patching_bridge_from_config` is imported twice (lines 128-130 and 131-133). No functional impact.

### BUG-08: `rv_l27_activation_patching_bridge.py` Hardcodes Mistral Architecture (MEDIUM SEVERITY)

**File**: `src/pipelines/canonical/rv_l27_activation_patching_bridge.py`, lines 203-207
**Issue**: Hook registration uses `model.model.layers[...].self_attn.v_proj` directly, bypassing the architecture-agnostic `hf_accessors.py` layer. This means this pipeline ONLY works with Llama/Mistral-style models. Pythia, GPT-2, OPT would crash.
**Impact**: The causal validation pipeline (`rv_l27_causal_validation.py`) correctly uses `hf_accessors.py`, but the activation patching bridge does not. This limits cross-architecture reproducibility.

### BUG-09: Confound Validation Uses `compute_rv()` Which Runs Model Internally (INCONSISTENCY)

**File**: `src/pipelines/canonical/confound_validation.py`, line 154
**Issue**: Uses `compute_rv(model, tokenizer, text, ...)` from `src.metrics.rv`, which internally captures V at both layers and computes PR. This is the self-contained version. Meanwhile, `rv_l27_causal_validation.py` uses `participation_ratio()` directly on pre-captured tensors. Both are correct but different code paths.
**Impact**: Different code paths for the same metric means different bugs could lurk in different experiments.

### BUG-10: Missing `configs/` Directory for Canonical Experiments

**Observation**: Only ONE canonical config file exists: `configs/canonical/rv_l27_causal_validation.json`. For the other 9 canonical experiments listed in the registry, there are scattered configs but not a complete set in `configs/canonical/`.
**Impact**: A reviewer cannot run `python -m src.pipelines.run --config configs/canonical/<experiment>.json` for most experiments. They would need to create configs from scratch.

---

## 3. Import Chain Map

### Which `participation_ratio` does each script import?

```
CANONICAL PIPELINES (src/pipelines/canonical/):
  rv_l27_causal_validation.py      -> src.metrics.rv.participation_ratio
  rv_l27_activation_patching_bridge.py -> src.metrics.rv.participation_ratio + compute_rv_with_components
  confound_validation.py           -> src.metrics.rv.compute_rv (uses PR internally)
  multi_token_bridge.py            -> src.metrics.rv.compute_rv_with_components (uses PR internally)
  head_ablation_validation.py      -> src.metrics.rv.participation_ratio
  mlp_sufficiency_test.py          -> src.metrics.rv.compute_rv (uses PR internally)
  mlp_combined_sufficiency_test.py -> src.metrics.rv.compute_rv (uses PR internally)
  mlp_ablation_necessity_prompt_pass.py -> src.metrics.rv.compute_rv_with_components
  random_direction_control.py      -> src.metrics.rv.compute_rv (uses PR internally)
  rv_l27_kv_patching_bridge.py     -> src.metrics.rv.compute_rv_with_components + participation_ratio

POWER-UP SCRIPT:
  scripts/power_up_multiseed.py    -> geometric_lens.probe.GeometricProbe
                                      -> geometric_lens.metrics.participation_ratio (DIFFERENT IMPL)

VALIDATED PATCHING SCRIPT:
  archive/rv_paper_code/VALIDATED_*.py -> compute_metrics_fast() (THIRD IMPL, inline, float32, no transpose)

GEOMETRIC_LENS PROBE:
  geometric_lens/probe.py          -> geometric_lens.metrics.participation_ratio
```

### Summary

ALL canonical pipelines use `src.metrics.rv.participation_ratio` (GPU float64 SVD, no NaN guard).
The power-up multi-seed script uses `geometric_lens.metrics.participation_ratio` (CPU float64 SVD, NaN guard).
The validated patching script uses its own inline `compute_metrics_fast()` (GPU float32 SVD, allows short sequences).

**THREE distinct implementations** produce the R_V numbers reported in the paper. This is a reproducibility hazard.

---

## 4. Reproducibility Score: 5/10

### What works (strengths):

1. **Config-driven runner** (`src/pipelines/run.py`): Well-designed, creates run directories, saves configs, checksums, hardware info, git commit, and a JSONL ledger. This is publication-grade infrastructure.

2. **Prompt bank versioning**: SHA-256 hash of `prompts/bank.json` is stored with every run. This is excellent for reproducibility.

3. **Seed management**: `set_seed()` in `src/core/models.py` sets Python, NumPy, PyTorch, and CUDA seeds, enables deterministic algorithms. Good.

4. **Pinned dependencies**: `requirements.lock` exists with exact version pins. Good.

5. **Model architecture abstraction**: `src/core/hf_accessors.py` and `geometric_lens/models.py` both handle multiple architectures. Good.

6. **Control conditions**: The causal validation pipeline includes random, shuffled, and wrong-layer controls. Good experimental design.

### What does NOT work (weaknesses):

1. **Three PR implementations** (see Section 3). A reviewer running different scripts could get different numbers.

2. **No single entry point to reproduce ALL results**. There is no `reproduce_all.sh` or Makefile. The runner requires a config JSON, but most canonical configs don't exist. A reviewer would need to understand the entire config schema to write their own.

3. **Missing config files** for 9 of 10 canonical experiments (only `rv_l27_causal_validation.json` exists in `configs/canonical/`).

4. **Model version ambiguity**: The validated script uses Mistral-7B-Instruct-v0.2, the canonical pipeline defaults to Mistral-7B-v0.1 Base. Which results are in the paper?

5. **Qwen layer count bug** in the registry could silently produce wrong results for Qwen experiments.

6. **No tests**. There are zero unit tests for `participation_ratio`, `compute_rv`, or any pipeline function. No `tests/` directory.

7. **No `pyproject.toml` or `setup.py`** at the project root. The project cannot be `pip install -e .`'d. Import paths assume running from the project root with `PYTHONPATH=.`.

8. **`prompts/bank.json` is not in version control** (754 prompts, referenced as canonical source, but only its hash is saved). If the file is modified, old results cannot be reproduced.

9. **The power-up multi-seed script** (`scripts/power_up_multiseed.py`) uses its own inline prompt lists (100 recursive + 100 baseline), NOT the canonical `prompts/bank.json`. This means multi-seed results are on DIFFERENT prompts than the canonical pipeline results.

---

## 5. What's Needed for a NeurIPS Reproducibility Appendix

### P0 (Must-Have Before Submission):

1. **Unify PR implementation**: Choose ONE `participation_ratio()` function. Recommend `geometric_lens/metrics.py` (CPU float64, NaN guard). Make `src/metrics/rv.py` import from it or vice versa. Remove the inline version in the validated script.

2. **Fix Qwen registry**: Change `num_layers` from 32 to 28 for `Qwen/Qwen2.5-7B`. Update `late_layer` from 27 to 23.

3. **Resolve model version**: Decide whether the paper reports results on Mistral-7B-v0.1 (Base) or Mistral-7B-Instruct-v0.2. Update all code and configs to be consistent. Document the choice.

4. **Create all canonical config JSONs**: One JSON per canonical experiment, stored in `configs/canonical/`. These must be the exact configs that produce the paper's numbers.

5. **Create `reproduce_all.sh`**: A single script that runs all experiments in sequence and produces all figures. Must work from a fresh clone.

6. **Add `pyproject.toml`**: So the project can be installed as a package. This fixes the import path fragility.

7. **Version-pin `prompts/bank.json`**: Either include it in the repo (it's text, not large) or provide a download URL with checksum.

### P1 (Strongly Recommended):

8. **Add unit tests**: At minimum, test `participation_ratio` with known inputs (identity matrix, rank-1 matrix, random matrix) and verify against hand-computed values.

9. **Unify the prompt bank**: The power-up script's inline prompts should be replaced with `PromptLoader` calls to ensure all experiments use the same prompts.

10. **Document the measurement contract**: Create a one-page document that specifies: "R_V is computed as PR(V_late)/PR(V_early) where PR = (sum sigma_i^2)^2 / sum(sigma_i^4), SVD is computed in float64 on CPU, window = last 16 tokens, early = layer 5, late = layer 27 (Mistral) or auto-detected at 84% depth." This should be in the paper's methods section AND as a docstring.

11. **Fix double forward pass**: Use `capture_multi_layer` to capture both early and late V-projections in a single forward pass. This halves computation time and eliminates any theoretical concern about model state changing between passes (it shouldn't, since model is in eval mode with no_grad, but a reviewer might ask).

### P2 (Nice to Have):

12. **Add a `Dockerfile`** or `environment.yml` for exact environment reproduction.
13. **Add CI** that runs at least the unit tests.
14. **Remove the `archive/` directory** from the submission (or clearly mark it as non-essential). Having 90+ archived scripts is overwhelming for a reviewer.

---

## 6. Critical Code Fixes Needed Before Submission

### FIX-01: Unify `participation_ratio` (CRITICAL)

Create a single authoritative implementation. Recommended: use the `geometric_lens/metrics.py` version (CPU, float64, NaN/Inf guard). Then:

```python
# In src/metrics/rv.py, replace the function with:
from geometric_lens.metrics import participation_ratio
```

Or vice versa. The point is: ONE function, imported everywhere.

### FIX-02: Fix Qwen Registry (HIGH)

In `geometric_lens/models.py`, line 217-222:
```python
# CURRENT (WRONG):
"Qwen/Qwen2.5-7B": ModelSpec(
    name="Qwen/Qwen2.5-7B",
    num_layers=32, ...
    early_layer=5, late_layer=27,
),

# FIXED:
"Qwen/Qwen2.5-7B": ModelSpec(
    name="Qwen/Qwen2.5-7B",
    num_layers=28, num_heads=28, num_kv_heads=4,
    hidden_size=3584, head_dim=128, proj_kind="separate",
    early_layer=4, late_layer=23,
),
```

### FIX-03: Remove Dead Code (LOW)

In `src/metrics/rv.py`, remove line 86:
```python
p = S_sq / total_variance  # Normalized eigenvalues  # DELETE THIS LINE
```

### FIX-04: Fix Architecture Hardcoding (MEDIUM)

In `src/pipelines/canonical/rv_l27_activation_patching_bridge.py`, lines 203-207, replace:
```python
layer_early = model.model.layers[early_layer].self_attn
layer_patch = model.model.layers[patch_layer].self_attn
h_early = layer_early.v_proj.register_forward_hook(hook_capture_early)
h_patch = layer_patch.v_proj.register_forward_hook(hook_patch_and_capture)
```
with architecture-agnostic accessors from `src.core.hf_accessors`.

### FIX-05: Remove Duplicate Import (COSMETIC)

In `src/pipelines/registry.py`, remove the duplicate import at lines 131-133:
```python
from .canonical.rv_l27_kv_patching_bridge import (
    run_rv_l27_kv_patching_bridge_from_config,
)
```

### FIX-06: Resolve Model Version (HIGH)

In `archive/rv_paper_code/VALIDATED_mistral7b_layer27_activation_patching.py`, either:
- Change the docstring to reference `Mistral-7B-v0.1` (Base), OR
- Change `src/core/models.py` default to `Mistral-7B-Instruct-v0.2`
- Ensure the paper clearly states which model was used

---

## Summary Table

| Category | Status | Details |
|---|---|---|
| PR formula correctness | PASS | All three implementations compute the same formula |
| PR numerical consistency | PARTIAL | float32 vs float64, GPU vs CPU can produce small differences |
| Model registry accuracy | FAIL | Qwen2.5-7B has wrong layer count |
| Prompt bank integrity | PASS | SHA-256 versioned, canonical source exists |
| Single entry point | FAIL | No `reproduce_all.sh`, missing config files |
| Dependency pinning | PASS | `requirements.lock` exists |
| Unit tests | FAIL | Zero tests |
| Code deduplication | FAIL | Three PR implementations |
| Architecture portability | PARTIAL | Some pipelines hardcode Mistral paths |
| Documentation | PARTIAL | Good docstrings, missing methods spec |

**Overall Reproducibility Score: 5/10**

A competent reviewer with PyTorch experience could probably reproduce the core results (R_V causal validation on Mistral-7B) with moderate effort (2-4 hours of reading code and creating configs). But reproducing ALL results across ALL models would be very difficult without the fixes above. For NeurIPS, this needs to be at least 7/10.

---

*Audit conducted by examining all source files in the repository. No code was executed.*
