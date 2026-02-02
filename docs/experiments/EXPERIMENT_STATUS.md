# Circuit Hunt V2: Experiment Status

## Current Status: ✅ RUNNING

**Process ID:** 11164 (parent) / 11170 (python)  
**Started:** ~1 minute ago  
**Status:** Loading model and prompts (initialization phase)

## What's Happening

The experiment is currently in the initialization phase:
1. ✅ Process started
2. ⏳ Loading Mistral-7B-v0.1 model (~30-60 seconds)
3. ⏳ Loading prompts from prompt bank
4. ⏳ Starting Experiment 1: Early-Layer Head Ablation

## Expected Timeline

- **Initialization:** 1-2 minutes (model loading)
- **Experiment 1 (Early-Layer Head Ablation):** ~1-2 hours
  - 6 layers × 32 heads × 8 prompts = ~1,536 forward passes
- **Experiment 2 (Mean Ablation):** ~30-45 minutes
  - 3 layers × 16 heads × 8 prompts = ~384 forward passes
- **Experiment 3 (Reverse Patching):** ~15-30 minutes
  - 9 layers × 8 prompt pairs = ~72 forward passes

**Total Estimated Time:** 2-4 hours

## Monitor Progress

Run this command to check status:
```bash
./monitor_experiment.sh
```

Or check results directory:
```bash
ls -lth results/circuit_hunt_v2_focused/
```

## What to Expect

The experiment will:
1. Test all 32 heads at layers 10, 12, 15, 18, 20, 22 (ramp region)
2. Compare mean ablation vs zeroing at L15, L20, L27
3. Test reverse patching (baseline → recursive) at all ramp + late layers

## Results Location

Results will be saved to:
```
results/circuit_hunt_v2_focused/results_YYYYMMDD_HHMMSS.json
```

The script will print a summary of significant effects when complete.

## Success Indicators

Watch for:
- **Head ablation effects:** |Δ| > 0.02
- **Mean ablation differences:** > 0.02 between mean and zero
- **Reverse patching recovery:** > 20% recovery toward baseline

## If Process Stops

If the process stops unexpectedly:
1. Check for error messages in terminal
2. Check GPU memory: `nvidia-smi`
3. Check disk space: `df -h`
4. Review logs if available

---

**Experiment is running. The GPU is working. Let's find that circuit!** 🔬
