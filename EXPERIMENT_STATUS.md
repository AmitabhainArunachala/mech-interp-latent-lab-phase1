# NeurIPS n=300 Experiment Status

## Experiment Running

**Status:** ✅ Running in background  
**PID:** Check `neurips_n300.pid`  
**Log:** `neurips_n300_output.log`

## What's Running

- **Script:** `neurips_n300_robust_experiment.py`
- **N:** 300 prompt pairs
- **Method:** Full KV cache + Persistent V_PROJ at L27
- **Controls:** Baseline, Random, Wrong Layer
- **Metrics:** Behavior score + R_V

## Expected Runtime

- **Per pair:** ~10-15 seconds (4 conditions × ~3s each)
- **Total:** ~50-75 minutes for 300 pairs

## Monitor Progress

```bash
# Check if running
ps aux | grep neurips_n300

# Check progress
tail -f neurips_n300_output.log | grep "Processing pairs"

# Check results (when done)
cat neurips_n300_results.csv | wc -l  # Should be 301 (header + 300 rows)
```

## Output Files

- `neurips_n300_results.csv` - Full pair-level results
- `neurips_n300_summary.md` - Statistical summary
- `neurips_n300_output.log` - Execution log

## Expected Results

Based on pilot (n=1):
- **Transfer:** Behavior score ~11 (baseline ~0)
- **Transfer efficiency:** ~100%
- **Controls:** Minimal/no effect

## Next Steps

Once complete:
1. Review statistical summary
2. Generate figures/plots
3. Write paper section
4. Prepare supplementary materials

