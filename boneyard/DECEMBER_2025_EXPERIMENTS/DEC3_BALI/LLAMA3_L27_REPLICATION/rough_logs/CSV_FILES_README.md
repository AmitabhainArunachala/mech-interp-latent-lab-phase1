# CSV Results Files - Expected Location

## Files Generated During December 3, 2025 Session

The following CSV files should be copied to this `rough_logs/` folder from the RunPod results directory:

### Llama-3-8B Results

1. **llama3_L27_FULL_VALIDATION_20251203_054646.csv**
   - Layer 27 attempt (initial replication)
   - n=45 pairs
   - Transfer efficiency: 1625% (artifact due to tiny natural gap)

2. **llama3_L27_FULL_VALIDATION_20251203_065527.csv**
   - Layer 24 validation (optimal layer)
   - n=45 pairs
   - Transfer efficiency: 271.2%
   - Main effect: Δ=-0.2091, d=-2.329

### Mistral-7B Results

3. **mistral_L20_FULL_VALIDATION_20251203_072103.csv**
   - Layer 20 attempt (from layer sweep)
   - n=30 pairs
   - Part of layer discovery process

4. **mistral_L22_FULL_VALIDATION_20251203_073538.csv**
   - Layer 22 validation (optimal layer)
   - n=30 pairs
   - Transfer efficiency: 119.7%
   - Main effect: Δ=-0.0797, d=-1.213

---

## Expected CSV Columns

Each CSV should contain:
- `rec_id` - Recursive prompt ID
- `base_id` - Baseline prompt ID
- `rec_group` - Recursive group (L3_deeper, L4_full, L5_refined)
- `base_group` - Baseline group (long_control, baseline_creative, baseline_math)
- `RV{layer}_rec` - R_V for recursive prompt
- `RV{layer}_base` - R_V for baseline prompt
- `RV{layer}_patch_main` - R_V after patching (main condition)
- `delta_main` - Main effect (patched - baseline)
- `delta_random` - Random control delta
- `delta_shuffled` - Shuffled control delta
- `delta_wronglayer` - Wrong-layer control delta

---

## To Copy Files

If files are on RunPod, use:
```bash
# From RunPod terminal
scp results/*.csv user@local:/path/to/LLAMA3_L27_REPLICATION/rough_logs/
```

Or download via RunPod web interface and place in this folder.

---

*Last updated: December 3, 2025*

