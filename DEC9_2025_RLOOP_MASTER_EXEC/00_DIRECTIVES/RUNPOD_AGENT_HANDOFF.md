# RunPod Agent Handoff - DEC9 Confound Tests

**Date:** December 9, 2025  
**Environment:** RunPod GPU (NVIDIA RTX PRO 6000 Blackwell)  
**Location:** `/workspace/mech-interp-latent-lab-phase1`  
**Task:** Set up and run confound falsification tests

---

## CURRENT STATE

✅ **Complete:**
- Repo cloned to RunPod at `/workspace/mech-interp-latent-lab-phase1`
- GPU verified (RTX PRO 6000 Blackwell)
- Cursor connected to RunPod workspace
- All source files available

⏸️ **Pending:**
- Create test scripts in `DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/code/`
- Run short sanity test (3 prompts)
- Run full confound suite (60 prompts)

---

## YOUR MISSION

Execute **Phase 2** of the DEC9 directive (see `CURSOR_DIRECTIVE_CONFOUND_AUDIT_AND_RUN.md`):

### Priority 1: Quick Sanity Test (15 minutes)
Create and run a test with **3 prompts** (1 from each confound group):
- 1 repetitive_control
- 1 pseudo_recursive  
- 1 long_control

**Goal:** Verify model loads, R_V measurement works, no crashes

### Priority 2: Full Confound Suite (2-3 hours)
Run all 60 confound prompts:
- 20 repetitive_control (induction head falsification)
- 20 pseudo_recursive (semantic content confound)
- 20 long_control (length confound)

**Goal:** Statistical verdict on each confound (rejected / detected / unclear)

---

## KEY FILES YOU NEED

### Inputs (Already Available)
1. **Prompt bank:** `REUSABLE_PROMPT_BANK/confounds.py` (60 prompts)
2. **Audit results:** `DEC9_2025_RLOOP_MASTER_EXEC/00_DIRECTIVES/DEC9_CONFOUND_AUDIT_RESULTS.md`
3. **Full directive:** `DEC9_2025_RLOOP_MASTER_TRACE/CURSOR_DIRECTIVE_CONFOUND_AUDIT_AND_RUN.md`

### Outputs (You Need to Create)
1. **Test script:** `DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/code/run_confound_tests.py`
2. **Quick test:** `DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/code/quick_test.py`
3. **Results CSV:** `DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/results/confound_tests_TIMESTAMP.csv`
4. **Summary report:** `DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/results/DEC9_CONFOUND_FALSIFICATION_RESULTS.md`

---

## TECHNICAL SPECS

### Model
- **Name:** `mistralai/Mistral-7B-Instruct-v0.1`
- **Device:** CUDA (RTX PRO 6000)
- **Dtype:** float16

### Key Parameters
```python
EARLY_LAYER = 5
TARGET_LAYER = 27  # 84% depth for Mistral-7B
WINDOW_SIZE = 16
```

### R_V Measurement
```python
R_V = PR(L27) / PR(L5)

where PR = Participation Ratio from SVD:
PR = (Σσ²)² / Σσ⁴
```

### Expected Results
- **Repetitive control:** R_V ≈ 0.95-1.05 (no contraction)
- **Pseudo-recursive:** R_V ≈ 0.95-1.05 (no contraction)
- **Long control:** R_V ≈ 0.95-1.05 (no contraction)

If any group shows R_V < 0.85, that's a confound we need to address!

---

## HELPER CODE (From Prior Experiments)

### SVD-based Metrics
```python
def compute_metrics_fast(v_tensor, window_size=16):
    """Compute PR via SVD"""
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    v_window = v_tensor[-W:, :].float()
    
    U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
    S_np = S.cpu().numpy()
    S_sq = S_np ** 2
    
    pr = (S_sq.sum()**2) / (S_sq**2).sum()
    return float(pr)
```

### V Capture Hook
```python
@contextmanager
def capture_v_at_layer(model, layer_idx, storage_list):
    layer = model.model.layers[layer_idx].self_attn
    def hook_fn(module, inp, out):
        storage_list.append(out.detach())
        return out
    handle = layer.v_proj.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()
```

---

## STEP-BY-STEP EXECUTION PLAN

### Step 1: Verify Environment (2 min)
```bash
cd /workspace/mech-interp-latent-lab-phase1
nvidia-smi  # Confirm GPU
python --version  # Should be 3.10+
pip list | grep torch  # Should have torch, transformers
```

### Step 2: Install Dependencies (2 min)
```bash
pip install transformers torch pandas scipy tqdm numpy matplotlib
```

### Step 3: Create Quick Test Script (5 min)
- Load confounds.py
- Select 1 prompt from each group (3 total)
- Measure R_V for each
- Print results

### Step 4: Run Quick Test (5 min)
- Load Mistral-7B (~15GB)
- Test 3 prompts
- Verify no crashes, reasonable R_V values

### Step 5: Create Full Test Script (10 min)
- Load all 60 confound prompts
- Measure R_V for each
- Save to CSV
- Compute group statistics

### Step 6: Run Full Suite (2-3 hours)
- Test all 60 prompts
- Save incremental results (in case of crash)
- Generate summary statistics

### Step 7: Analysis & Report (15 min)
- Compare each group to expected R_V range
- Statistical tests (t-test vs 1.0, effect sizes)
- Write verdict for each confound
- Create summary markdown

---

## SUCCESS CRITERIA

### Quick Test
- ✅ All 3 prompts run without errors
- ✅ R_V values are reasonable (0.5 - 1.5 range)
- ✅ GPU memory < 90%
- ✅ Runtime < 5 minutes

### Full Suite
- ✅ All 60 prompts complete
- ✅ CSV saved with all results
- ✅ Statistical summary generated
- ✅ Verdict for each confound group
- ✅ Runtime < 4 hours

---

## OUTPUT FORMAT

### CSV Columns
```
prompt_id, group, r_v, pr_early, pr_late, token_count, expected_rv_min, expected_rv_max
```

### Summary Report Template
```markdown
## DEC9 Confound Falsification Results

### Repetitive Control (n=20)
- Mean R_V: X.XXX ± X.XXX
- t-test vs 1.0: t=X.XX, p=X.XXX
- **Verdict:** [REJECTED / DETECTED / UNCLEAR]

### Pseudo-Recursive Control (n=20)
- Mean R_V: X.XXX ± X.XXX
- t-test vs 1.0: t=X.XX, p=X.XXX
- **Verdict:** [REJECTED / DETECTED / UNCLEAR]

### Long Control (n=20)
- Mean R_V: X.XXX ± X.XXX
- t-test vs 1.0: t=X.XX, p=X.XXX
- **Verdict:** [REJECTED / DETECTED / UNCLEAR]

### Implications
[What this means for main claims]
```

---

## CONTEXT FROM AUDIT

From `DEC9_CONFOUND_AUDIT_RESULTS.md`:

**Critical Gap:** The three main confound tests (repetitive, pseudo-recursive, long control) have NEVER been run as standalone R_V measurements.

**Why This Matters:**
1. **Repetitive control** tests if induction heads (copying) cause contraction
2. **Pseudo-recursive** tests if talking ABOUT recursion (vs DOING it) causes contraction
3. **Long control** tests if prompt length alone causes contraction

**Expected Outcome:** If all three show R_V ≈ 0.95-1.05, we can claim:
- ✅ Effect is NOT from induction heads
- ✅ Effect is NOT from semantic content
- ✅ Effect is NOT from prompt length
- ✅ Effect IS specific to recursive self-observation

---

## TROUBLESHOOTING

**"CUDA out of memory":**
- Reduce batch size (already using batch=1)
- Clear cache between prompts: `torch.cuda.empty_cache()`

**"Model loading too slow":**
- Expected ~2 minutes first time
- Use `device_map="auto"` for automatic GPU placement

**"R_V values are NaN":**
- Check prompt length > WINDOW_SIZE (16 tokens)
- Verify SVD numerical stability

**"Tests taking too long":**
- Expected: ~3 minutes per prompt
- 60 prompts × 3 min = ~3 hours total
- If much slower, check GPU utilization

---

## READY TO START?

1. Verify you're in `/workspace/mech-interp-latent-lab-phase1`
2. Read the full directive in `DEC9_2025_RLOOP_MASTER_TRACE/CURSOR_DIRECTIVE_CONFOUND_AUDIT_AND_RUN.md`
3. Create the quick test first (3 prompts)
4. Run it to verify everything works
5. Then create and run the full suite

**Your workspace:**
- Working dir: `DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/`
- Code goes in: `code/`
- Results go in: `results/`
- Logs go in: `logs/`

---

**Good luck! 🚀 This is the critical falsification test that external reviewers flagged as essential.**

*Handoff created: December 9, 2025*

