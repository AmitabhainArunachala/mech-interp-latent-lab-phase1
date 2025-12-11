# LLAMA-3-8B L27 REPLICATION CHECKLIST

## PRE-FLIGHT (Before any code runs)

### Environment
- [ ] RunPod instance running
- [ ] GPU confirmed (run `nvidia-smi`)
- [ ] GPU type: _____________ (e.g., A100, A6000)
- [ ] VRAM available: _______ GB

### Dependencies
- [ ] torch installed (version: _______)
- [ ] transformers installed (version: _______)
- [ ] scipy installed
- [ ] pandas installed
- [ ] numpy installed
- [ ] tqdm installed

### Files
- [ ] `llama3_L27_FULL_VALIDATION.py` created
- [ ] `n300_mistral_test_prompt_bank.py` accessible
- [ ] `prompt_bank_1c` loads without error

---

## MODEL LOADING

### Verification
- [ ] Model loads without OOM error
- [ ] Layer count verified: _______ (expected: 32)
- [ ] Layer 27 has v_proj: YES / NO
- [ ] Hook registration test passed: YES / NO

### Record
- [ ] Model name: `meta-llama/Meta-Llama-3-8B-Instruct`
- [ ] Dtype: _______ (expected: float16)
- [ ] Device: _______ (expected: cuda)
- [ ] Load time: _______ seconds
- [ ] VRAM after load: _______ GB

---

## SANITY CHECK (1 prompt pair)

### Run
- [ ] Single recursive prompt processed
- [ ] Single baseline prompt processed
- [ ] V tensors captured at Layer 5: shape = _______
- [ ] V tensors captured at Layer 27: shape = _______

### Metrics
- [ ] PR(early) computes without error: _______
- [ ] PR(late) computes without error: _______
- [ ] R_V computes: _______
- [ ] Patching runs without error: YES / NO

### Sanity values (record raw)
- Recursive R_V: _______
- Baseline R_V: _______
- Patched R_V: _______
- Delta: _______

---

## FULL VALIDATION (n=45)

### Run parameters
- [ ] Start time: _______
- [ ] Prompt pairs loaded: _______ (expected: 45)
- [ ] Random seed set: 42

### Progress
- [ ] 10/45 complete
- [ ] 25/45 complete
- [ ] 45/45 complete
- [ ] End time: _______
- [ ] Total runtime: _______ minutes

### Errors
- [ ] Any OOM errors? YES / NO (if yes, note which pair)
- [ ] Any NaN values? YES / NO
- [ ] Any pairs skipped? YES / NO (count: ___)

---

## RESULTS

### Main effect
- [ ] Mean delta (main): _______
- [ ] Std delta (main): _______
- [ ] t-statistic: _______
- [ ] p-value: _______
- [ ] Cohen's d: _______

### Transfer efficiency
- [ ] Natural gap (baseline - recursive): _______
- [ ] Transfer efficiency: _______% 

### Controls
- [ ] Random delta: _______ (expected: positive/opposite)
- [ ] Shuffled delta: _______ (expected: reduced)
- [ ] Wrong-layer delta: _______ (expected: ~0)

### Control p-values
- [ ] Main vs Random: p = _______
- [ ] Main vs Shuffled: p = _______
- [ ] Wrong-layer vs 0: p = _______

---

## COMPARISON TO MISTRAL

| Metric | Mistral-7B | Llama-3-8B | Difference |
|--------|------------|------------|------------|
| Transfer % | 117.8% | _______ | _______ |
| Cohen's d | -3.56 | _______ | _______ |
| p-value | <10⁻⁶ | _______ | _______ |
| Random ctrl | +0.711 | _______ | _______ |
| Wrong-layer | +0.000 | _______ | _______ |

---

## VERDICT

- [ ] Transfer >50%: UNIVERSAL MECHANISM CONFIRMED
- [ ] Transfer 20-50%: PARTIAL REPLICATION
- [ ] Transfer ~0%: ARCHITECTURE-SPECIFIC

## Notes
_______________________________________
_______________________________________
_______________________________________

## Files saved
- [ ] CSV: `results/llama3_L27_validation_YYYYMMDD_HHMMSS.csv`
- [ ] Log: `logs/run_log_YYYYMMDD_HHMMSS.txt`

