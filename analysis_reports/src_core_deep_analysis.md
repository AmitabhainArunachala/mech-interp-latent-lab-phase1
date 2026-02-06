# src/core/ Module Deep Analysis Report

## Executive Summary

The `src/core/` directory contains **1,278 lines** of core utilities across 9 Python files. This analysis reveals significant functional overlap with `rv_toolkit/`, inconsistent abstractions, and several architectural issues.

---

## File-by-File Analysis

### 1. models.py (104 lines)

**Complexity: LOW** | **Dependencies: transformers, torch**

**Key Functions:**
| Function | Purpose | Lines |
|----------|---------|-------|
| `set_seed()` | Reproducibility with CUDA determinism | 35 |
| `load_model()` | HF model/tokenizer loading with fallback | 60 |

**Issues Identified:**
1. **Missing numpy import handling** - Uses try/except for numpy but not consistently
2. **Hardcoded defaults** - Mistral-7B hardcoded as "reference reality"
3. **No caching support** - Doesn't integrate with model_physics registry
4. **Incomplete docstring** - `attn_implementation` param documented but not fully validated

**Duplications with rv_toolkit:**
- NONE directly, but rv_toolkit experiments duplicate model loading logic inline

---

### 2. hooks.py (175 lines)

**Complexity: MEDIUM** | **Dependencies: torch, transformers**

**Key Functions:**
| Function | Purpose | Lines |
|----------|---------|-------|
| `capture_v_projection()` | Context manager for V_PROJ hook | 38 |
| `capture_attention_patterns()` | Capture attention weights | 36 |
| `capture_head_output()` | Extract single head contribution | 45 |
| `capture_hidden_states()` | Capture layer hidden states | 32 |

**Issues Identified:**
1. **Architecture-specific hardcoding** - Assumes `model.model.layers[layer_idx].self_attn` structure (Mistral/LLaMA only)
2. **No GPT-2 support** - rv_toolkit's patcher handles multiple architectures, hooks.py doesn't
3. **Inconsistent return patterns** - Some yield dicts, some use storage_list param
4. **Missing error handling** - No validation for layer_idx bounds

**Duplications with rv_toolkit:**
- rv_toolkit/patching.py has `_get_v_proj()` and `_get_v_tensor()` with similar functionality
- rv_toolkit/metrics.py has inline hook registration in `compute_rv_layerwise()`

---

### 3. patching.py (313 lines)

**Complexity: HIGH** | **Dependencies: torch**

**Key Classes/Functions:**
| Symbol | Purpose | Lines |
|--------|---------|-------|
| `PersistentVPatcher` | V_PROJ patching during generation | 96 |
| `PersistentResidualPatcher` | Residual stream patching | 96 |
| `extract_v_activation()` | Extract V tensor from prompt | 43 |
| `extract_residual_activation()` | Extract residual tensor | 42 |

**Issues Identified:**
1. **CRITICAL: Hardcoded window_size=16** - Magic number repeated in both classes
2. **CRITICAL: Hardcoded Mistral dimensions** - `num_heads=32`, `head_dim=128`, `hidden_dim=4096`
3. **No architecture detection** - rv_toolkit's `ActivationPatcher` has `_detect_architecture()`
4. **Duplicated logic** - Both patcher classes have nearly identical `register()`/`remove()`/`__exit__` implementations
5. **Missing input validation** - No checks for valid layer_idx or tensor shapes
6. **Inconsistent batch handling** - Some places squeeze, others don't

**Duplications with rv_toolkit:**
| src/core/ | rv_toolkit/ | Overlap |
|-----------|-------------|---------|
| `PersistentVPatcher` | `ActivationPatcher` | ~60% - same patching concept |
| `extract_v_activation()` | `_get_v_tensor()` | ~80% - nearly identical |
| `extract_residual_activation()` | None | NEW |
| Window size logic | `window_size` param | Same magic number |

**Refactoring Opportunity:**
The two `Persistent*Patcher` classes should share a base class with common `register()`/`remove()`/`__exit__` logic.

---

### 4. logit_capture.py (189 lines)

**Complexity: MEDIUM** | **Dependencies: torch, transformers**

**Key Functions:**
| Function | Purpose | Lines |
|----------|---------|-------|
| `capture_logits()` | Context manager for lm_head hook | 63 |
| `capture_logits_during_generation()` | Step-by-step logit capture | 67 |
| `extract_logits_from_outputs()` | Output format normalizer | 34 |

**Issues Identified:**
1. **Duplicated hook registration logic** - Same pattern as hooks.py
2. **Inconsistent attribute checking** - Checks `lm_head`, `embed_out`, `head` but not systematically
3. **Missing tests** - No validation that hooks actually capture
4. **Unused parameter** - `max_steps` in `capture_logits_during_generation` not properly implemented
5. **No integration** - Not used by other core modules

**Duplications with rv_toolkit:**
- NONE - this is unique to src/core/
- However, rv_toolkit experiments likely inline similar logic

---

### 5. model_physics.py (101 lines)

**Complexity: LOW** | **Dependencies: dataclasses, typing**

**Key Components:**
| Symbol | Purpose | Lines |
|--------|---------|-------|
| `ModelPhysics` | Dataclass for model constants | 23 |
| `_REGISTRY` | Hardcoded model configs | 35 |
| `get_model_physics()` | Registry accessor with fallback | 23 |

**Issues Identified:**
1. **Hardcoded head indices** - `suppressor_heads=[(27, 18), (27, 26)]` without explanation
2. **Incomplete registry** - Only 4 models, most with empty head lists
3. **Print instead of log** - Warning uses `print()` instead of proper logging
4. **No validation** - Doesn't verify model configs against actual models
5. **Unused by other modules** - Not imported by patching.py, hooks.py, etc.

**Duplications with rv_toolkit:**
- rv_toolkit/patching.py has `_detect_architecture()` which overlaps conceptually

---

### 6. head_specific_patching.py (262 lines)

**Complexity: HIGH** | **Dependencies: torch**

**Key Classes:**
| Class | Purpose | Lines |
|-------|---------|-------|
| `HeadSpecificVPatcher` | Patch specific attention heads | 122 |
| `HeadSpecificSteeringPatcher` | Steering vector on heads | 118 |

**Issues Identified:**
1. **CRITICAL: Hardcoded Mistral dimensions** - Same issue as patching.py
2. **Code duplication** - Both classes duplicate dimension calculation logic
3. **No base class** - Should inherit from common patching base
4. **Hardcoded window_size=16** - Same magic number again
5. **Missing validation** - No check that target_heads are valid indices
6. **Architecture assumption** - Assumes `model.model.layers[layer_idx].self_attn.v_proj`

**Duplications with rv_toolkit:**
- NONE directly - this is novel functionality
- However, architectural issues mirror patching.py

**Relationship to patching.py:**
These classes should share a base with `PersistentVPatcher` but don't. The head dimension slicing logic is duplicated.

---

### 7. experiment_io.py (76 lines)

**Complexity: LOW** | **Dependencies: pathlib, json, datetime**

**Key Functions:**
| Function | Purpose | Lines |
|----------|---------|-------|
| `create_run_dir()` | Timestamped run directory | 28 |
| `write_json()` | JSON file writer | 12 |
| `write_text()` | Text file writer | 9 |
| `atomic_config_snapshot()` | Config persistence | 10 |

**Issues Identified:**
1. **Not used by other core modules** - Completely isolated
2. **Missing features** - No CSV writer, no result aggregation
3. **No error handling** - `mkdir()` failures not caught
4. **Simple wrappers** - These are thin wrappers around stdlib

**Duplications with rv_toolkit:**
- rv_toolkit experiments likely have their own I/O logic

---

### 8. utils.py (32 lines)

**Complexity: VERY LOW** | **Dependencies: NONE**

**Key Functions:**
| Function | Purpose | Lines |
|----------|---------|-------|
| `behavior_score()` | Keyword counting for recursion | 15 |

**Issues Identified:**
1. **Naive implementation** - Simple substring matching, no NLP
2. **Hardcoded keyword list** - `RECURSIVE_KEYWORDS` not configurable
3. **No validation** - Empty text returns 0 (might be unexpected)
4. **Overlapping matches** - "self-aware" counts as 2 but should be 1

**Duplications with rv_toolkit:**
- rv_toolkit/prompts.py has `RECURSIVE_PROMPTS` and `BASELINE_PROMPTS`
- This is related but not duplicated functionality

---

### 9. __init__.py (26 lines)

**Purpose:** Package exports

**Exports:**
- `load_model`, `set_seed`
- `capture_v_projection`, `capture_hidden_states`
- `PersistentVPatcher`, `PersistentResidualPatcher`
- `extract_v_activation`, `extract_residual_activation`
- `behavior_score`

**Issues:**
1. **Incomplete exports** - `capture_attention_patterns`, `capture_head_output` missing
2. **No `logit_capture` exports** - Module is isolated
3. **No `head_specific_patching` exports** - Classes not exposed
4. **No `model_physics` exports** - Registry not accessible
5. **No `experiment_io` exports** - I/O utilities hidden

---

## Cross-Module Dependency Analysis

```
models.py
    ↓ (used by)
    hooks.py, patching.py, head_specific_patching.py

hooks.py
    ↓ (used by)
    patching.py (similar patterns)
    ↓ (imports)
    NONE from core/

patching.py
    ↓ (used by)
    head_specific_patching.py (similar patterns)
    ↓ (imports)
    NONE from core/

head_specific_patching.py
    ↓ (imports)
    NONE from core/

model_physics.py
    ↓ (used by)
    NONE - completely isolated!

logit_capture.py
    ↓ (used by)
    NONE - completely isolated!

experiment_io.py
    ↓ (used by)
    NONE - completely isolated!

utils.py
    ↓ (used by)
    __init__.py only
```

**Key Finding:** Modules are highly decoupled but also poorly integrated. There's no shared base classes or common utilities.

---

## Duplications Summary: src/core/ vs rv_toolkit/

| src/core/ | rv_toolkit/ | Lines | Overlap |
|-----------|-------------|-------|---------|
| `patching.py:PersistentVPatcher` | `patching.py:ActivationPatcher` | ~200 | 60% |
| `patching.py:extract_v_activation()` | `patching.py:_get_v_tensor()` | ~40 | 80% |
| `hooks.py:capture_*` | `metrics.py:compute_rv_layerwise()` hooks | ~150 | 50% |
| `models.py:load_model()` | Inline in experiments | ~60 | 30% |
| Window size logic | Multiple locations | ~20 | 100% |

**Total duplicated logic: ~470 lines (37% of src/core/)**

---

## Critical Issues Found

### 🔴 HIGH SEVERITY

1. **Hardcoded Architecture Assumptions**
   - Assumes Mistral-7B structure throughout
   - `model.model.layers[layer_idx].self_attn.v_proj` pattern
   - No support for GPT-2, OPT, or other architectures
   - **Affected files:** hooks.py, patching.py, head_specific_patching.py

2. **Magic Numbers Everywhere**
   - `window_size = 16` repeated in 4+ locations
   - `num_heads = 32`, `head_dim = 128`, `hidden_dim = 4096` hardcoded
   - `layer_idx = 27` mentioned in docstrings but not validated
   - **Affected files:** patching.py, head_specific_patching.py

3. **No Shared Base Classes**
   - `PersistentVPatcher` and `PersistentResidualPatcher` ~90% similar
   - `HeadSpecificVPatcher` and `HeadSpecificSteeringPatcher` ~80% similar
   - DRY principle violated
   - **Affected files:** patching.py, head_specific_patching.py

### 🟡 MEDIUM SEVERITY

4. **Incomplete __init__.py Exports**
   - Several modules not exposed
   - Inconsistent public API
   - **File:** __init__.py

5. **Model Physics Registry Unused**
   - `model_physics.py` completely isolated
   - Other modules hardcode same values
   - **Files:** model_physics.py, all others

6. **No Architecture Detection**
   - rv_toolkit has `_detect_architecture()` method
   - src/core/ assumes Mistral structure
   - **Files:** hooks.py, patching.py, head_specific_patching.py

### 🟢 LOW SEVERITY

7. **Missing Input Validation**
   - No bounds checking on layer_idx
   - No tensor shape validation
   - **Files:** patching.py, head_specific_patching.py, hooks.py

8. **Inconsistent Error Handling**
   - Some places raise, others print
   - No standardized exception types
   - **Files:** All

9. **Unused Modules**
   - `logit_capture.py` not integrated
   - `experiment_io.py` not used
   - **Files:** logit_capture.py, experiment_io.py

---

## Recommendations

### Immediate (Phase 2.2)

1. **Create shared base class**
   ```python
   class BasePatcher(ABC):
       def register(self, layer_idx): ...
       def remove(self): ...
       def __enter__/__exit__(self): ...
   ```

2. **Centralize constants**
   - Use `model_physics.py` registry
   - Remove hardcoded dimensions

3. **Fix __init__.py exports**
   - Expose all public APIs
   - Document what's internal vs external

### Short-term (Phase 3)

4. **Merge with rv_toolkit**
   - Consolidate patching logic
   - Single source of truth for metrics

5. **Add architecture detection**
   - Port from rv_toolkit
   - Support GPT-2, LLaMA, Mistral

6. **Add comprehensive tests**
   - Current modules have no test coverage indicators

---

## Appendix: Line Counts Summary

| File | Lines | Category |
|------|-------|----------|
| patching.py | 313 | Patching |
| head_specific_patching.py | 262 | Patching |
| logit_capture.py | 189 | Capture |
| hooks.py | 175 | Capture |
| model_physics.py | 101 | Config |
| models.py | 104 | Loading |
| experiment_io.py | 76 | I/O |
| utils.py | 32 | Utilities |
| __init__.py | 26 | Exports |
| **TOTAL** | **1,278** | |

---

*Report generated: 2026-02-05*
*Analyst: subagent p2_src_core_analyzer*
