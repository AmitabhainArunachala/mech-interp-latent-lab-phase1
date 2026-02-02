# H31 Validation Setup: Honest Assessment

## Hardware Status

✅ **MPS Available:** Apple Silicon GPU acceleration works  
❌ **CUDA Available:** No NVIDIA GPU  
💾 **RAM:** 18 GB total (~14-15 GB usable after OS)

## Can We Run Locally?

### Model Requirements
- **Mistral-7B-v0.1:** ~7.24 GB model size
- **With float16:** ~14 GB RAM needed (model + activations)
- **With 8-bit quantization:** ~7 GB RAM needed

### Reality Check

**Option 1: MPS with float16**
- ✅ Should fit in 18 GB RAM
- ⚠️ **Will be SLOW** - Apple Silicon GPU is much slower than NVIDIA CUDA
- ⏱️ **Estimated runtime:** 1-2 hours for 50 prompts
- ⚠️ May have compatibility issues with `device_map="auto"`

**Option 2: CPU with float32**
- ✅ Will definitely work
- ❌ **VERY SLOW** - 4-6 hours for 50 prompts
- Not practical

**Option 3: Wait for RunPod**
- ✅ **RECOMMENDED**
- ⏱️ **Estimated runtime:** 10-15 minutes for 50 prompts
- ✅ Proper CUDA GPU acceleration
- ✅ No memory constraints

## Scripts Created

1. **`h31_validation_n50.py`** - Main validation script
   - Loads 50 recursive + 50 baseline prompts
   - Measures H31 BOS attention, entropy, R_V
   - Outputs CSV: `results/h31_validation/h31_validation_n50.csv`

2. **`test_model_load.py`** - Quick test script
   - Tests if model can load on this machine
   - Run this first to verify setup

## Recommendation

**Wait for RunPod.** Here's why:

1. **Speed:** 10-15 minutes vs 1-2 hours (MPS) or 4-6 hours (CPU)
2. **Reliability:** CUDA is well-tested, MPS may have issues
3. **Memory:** No constraints on RunPod GPU
4. **Compatibility:** All your existing code assumes CUDA

## If You Want to Try Local Anyway

```bash
# 1. Test model loading first
python3 test_model_load.py

# 2. If that works, run validation (will take 1-2 hours)
python3 h31_validation_n50.py
```

**Expected output:**
- CSV file: `results/h31_validation/h31_validation_n50.csv`
- Columns: prompt_type, prompt_text, rv, h31_entropy, h31_bos_attn, h31_max_attn, h31_marker_attn
- Summary printed to console

## What We're Validating

**Hypothesis:** H31 entropy perfectly separates recursive (0.28) vs baseline (0.81)

**Test:** Run on 50 recursive + 50 baseline prompts

**Success criteria:**
- Recursive mean entropy < 0.4
- Baseline mean entropy > 0.65
- Clear separation (no overlap)
- Statistical significance (t-test p < 0.001)

## Next Steps

1. **If RunPod available:** Run `h31_validation_n50.py` there (change device to "cuda")
2. **If local only:** Run `test_model_load.py` first, then decide
3. **Analysis:** Load CSV and compute statistics to validate the 0.28 vs 0.81 claim

---

**Bottom line:** The script is ready. It will work locally but be slow. RunPod is strongly recommended for speed and reliability.










