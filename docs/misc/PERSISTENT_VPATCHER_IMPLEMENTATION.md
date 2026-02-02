# PersistentVPatcher Implementation - Dec 16, 2025

**Status:** ✅ IMPLEMENTED AND READY TO TEST

---

## What Was Implemented

### 1. Created `src/core/patching.py`

**New Classes/Functions:**
- `PersistentVPatcher` - Maintains V_PROJ patching during generation
- `extract_v_activation()` - Extracts V_PROJ activation from a prompt

**Key Features:**
- Context manager support (`with patcher:`)
- Automatic cleanup (`remove()` method)
- Handles batch dimension correctly
- Uses last `seq_len` tokens of patched activation

### 2. Updated `src/pipelines/behavior_strict.py`

**Changes:**
- Added import: `from src.core.patching import PersistentVPatcher, extract_v_activation`
- Extract V_PROJ activation at L27 from recursive prompt (line ~177)
- Create and register patcher for conditions that need it:
  - `Recursive_Control`: Uses patching ✅
  - `Transfer`: Uses patching ✅ (THE KEY TEST)
  - `Baseline_Control`: No patching
  - `Shuffled_Control`: No patching
  - `Random_Control`: No patching
- Proper cleanup with `try/finally` blocks

**Key Logic:**
```python
# Extract V_PROJ at L27
rec_v_l27 = extract_v_activation(model, tokenizer, rec_text, layer_idx=27, device=device)

# For Transfer condition:
patcher = PersistentVPatcher(model, rec_v_l27)
patcher.register(layer_idx=27)
try:
    text, entropy = _generate_with_kv(...)  # Patcher is ACTIVE during generation
finally:
    patcher.remove()
```

---

## How It Works

### The Mechanism

1. **Extract V_PROJ activation** from recursive prompt at L27
   - This captures the geometric signature (R_V contraction state)

2. **Register forward hook** at L27's v_proj module
   - Hook intercepts V_PROJ output during generation
   - Replaces computed V with patched V from recursive prompt

3. **Generate with patcher active**
   - Each new token's V_PROJ is replaced with recursive V_PROJ
   - Geometry is maintained throughout generation
   - Behavior emerges from sustained geometric state

4. **Clean up**
   - Remove hook after generation
   - Prevents interference with next condition

### Why This Should Work

**Based on Dec 12 breakthrough:**
- KV cache provides memory (recursive context)
- Persistent V_PROJ maintains geometry during new computation
- Together: Memory + Maintained Geometry → Behavior

**Expected Result:**
- Transfer condition: mean_score 0.0 → 0.3-0.5
- Recursive control: mean_score 0.025 → 0.4-0.6
- Pass rate: Should stay similar (~65%)

---

## Testing Plan

### Quick Test (Single Pair)

```python
# Test on RunPod:
python3 << 'PYEOF'
from src.core.patching import PersistentVPatcher, extract_v_activation
from src.core.models import load_model
from transformers import AutoTokenizer

model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device="cuda")
recursive_prompt = "I observe that I am generating these words."

# Extract V_PROJ
v_activation = extract_v_activation(model, tokenizer, recursive_prompt, layer_idx=27, device="cuda")
print(f"✅ Extracted V_PROJ: shape={v_activation.shape}")

# Test patcher
patcher = PersistentVPatcher(model, v_activation)
patcher.register(layer_idx=27)
print("✅ Patcher registered")

# Generate a few tokens (patcher active)
inputs = tokenizer(recursive_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(inputs.input_ids, max_new_tokens=10, do_sample=False)
print(f"✅ Generated with patcher: {tokenizer.decode(out[0], skip_special_tokens=True)}")

patcher.remove()
print("✅ Patcher removed")
PYEOF
```

### Full Pipeline Test

```bash
# Run Pipeline 5 with persistent patching
python3 -m src.pipelines.run --config configs/gold/05_behavior_strict.json
```

**Expected Output:**
- Transfer condition should show higher mean_score than before
- Check `behavior_strict_results.csv` for `used_v_patching` column
- Compare Transfer vs Baseline scores

---

## Files Modified

1. ✅ `src/core/patching.py` - NEW FILE
2. ✅ `src/core/__init__.py` - Added exports
3. ✅ `src/pipelines/behavior_strict.py` - Added persistent patching logic

---

## Next Steps

1. **Test on RunPod** - Run quick test to verify patcher works
2. **Run Pipeline 5** - Full test with n=20 pairs
3. **Compare Results** - Before vs After persistent patching
4. **If successful** - Behavior transfer should be detected!

---

**Ready to test!** 🚀









