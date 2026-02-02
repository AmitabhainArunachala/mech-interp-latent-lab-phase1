# Running Circuit Hunt V2 Experiments

## Quick Start

### Option 1: Run on RunPod (Recommended)

1. **SSH into RunPod:**
```bash
ssh -p 18147 root@198.13.252.9
```

2. **Navigate to project directory:**
```bash
cd /workspace/mech-interp-latent-lab-phase1  # or wherever your project is
```

3. **Run the experiment:**
```bash
python3 experiment_circuit_hunt_v2_focused.py
```

Or use the helper script:
```bash
./run_circuit_hunt_on_runpod.sh
```

### Option 2: Quick Test First (Verify Setup)

Run a quick test to verify everything works:
```bash
python3 experiment_circuit_hunt_v2_quick_test.py
```

This tests just 3 heads at one layer to verify the code works.

## What Gets Tested

The focused version runs 3 experiments:

1. **Early-Layer Head Ablation** (L10-L22 ramp region)
   - Tests all 32 heads at 6 ramp layers
   - ~6 layers × 32 heads × 8 prompts = ~1,536 measurements

2. **Mean Ablation vs Zeroing** (L15, L20, L27)
   - Compares mean ablation vs zeroing
   - ~3 layers × 16 heads × 8 prompts = ~384 measurements

3. **Reverse Patching** (Baseline → Recursive)
   - Tests undoing the effect by patching baseline into recursive
   - ~9 layers × 8 prompt pairs = ~72 measurements

**Total:** ~2,000 forward passes (takes ~2-4 hours on GPU)

## Expected Output

Results are saved to:
```
results/circuit_hunt_v2_focused/results_YYYYMMDD_HHMMSS.json
```

The script prints a summary of significant effects:
- Head ablations with |Δ| > 0.02
- Mean ablation differences > 0.02
- Reverse patching with >20% recovery

## If You Find Significant Effects

If any experiment shows significant effects:

1. **Note the layer/head** that shows the effect
2. **Run targeted follow-up** experiments on that specific component
3. **Test interactions** - maybe it's pairs of heads, not singles

## If You Don't Find Anything

If all experiments still show no discrete circuit:

1. **Your conclusion is likely correct** - distributed, primarily MLPs
2. **But we've ruled out more hypotheses** - makes the paper stronger
3. **Document the negative results** - strengthens the distributed hypothesis

## Full Version (If Focused Shows Promise)

If the focused version shows promising results, run the full version:
```bash
python3 experiment_circuit_hunt_v2.py
```

This includes additional experiments:
- Head output patching
- Head interaction effects (pairs/triplets)
- Path patching

**Runtime:** ~8-12 hours on GPU

## Troubleshooting

### CUDA Out of Memory
- Reduce number of prompts in the script
- Process prompts in smaller batches
- Use gradient checkpointing (if needed)

### Slow Performance
- Make sure you're using GPU (check with `nvidia-smi`)
- Reduce number of test layers/heads
- Use the quick test version first

### Import Errors
- Make sure you're in the project root directory
- Check that all dependencies are installed
- Verify `prompts/loader.py` exists

## Next Steps After Running

1. **Analyze results** - Look for |Δ| > 0.02 or recovery > 20%
2. **If found:** Create targeted follow-up experiments
3. **If not found:** Document negative results and strengthen distributed hypothesis

---

**The goal:** Either find the circuit OR confirm with high confidence that it's distributed.

