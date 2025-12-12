# Unified Test Status

## Current Status

**Head-level ablation:** ✅ Complete
- Critical heads identified: L27 heads [11, 1, 22]
- Head 11: HIGH impact (6.1%)
- Heads 1, 22: MEDIUM impact (3.0%, 2.4%)

**Head activation extraction:** ✅ Complete
- Activations extracted and saved to `critical_heads_activations.npz`
- Summary saved to `critical_heads_summary.json`

**Unified test:** ⚠️ In Progress
- V_PROJ method working
- HEAD_LEVEL method has shape mismatch issue (needs debugging)
- RESIDUAL method has tuple handling issue (needs debugging)

## Next Steps

1. **Fix HEAD_LEVEL patching** - Debug shape mismatch in head activation patching
2. **Fix RESIDUAL patching** - Handle tuple outputs correctly
3. **Run full comparison** - Compare V_PROJ vs HEAD_LEVEL vs RESIDUAL

## Key Findings So Far

- **V_PROJ patching works** (confirmed from previous experiments: 86.5% transfer)
- **Head-level extraction works** (activations saved successfully)
- **Head-level patching needs debugging** (shape mismatch when patching)

## Files Generated

- `head_level_ablation.py` - Head ablation script
- `head_level_extraction.py` - Activation extraction script
- `unified_test_head_level.py` - Unified test script (in progress)
- `critical_heads_activations.npz` - Extracted activations
- `critical_heads_summary.json` - Metadata

## Recommendation

For now, **V_PROJ patching is proven** (86.5% transfer). The head-level patching is a refinement that will give cleaner signal, but V_PROJ already works well. We can:
1. Use V_PROJ for immediate experiments
2. Debug HEAD_LEVEL as a refinement
3. Focus on attention visualization of critical heads (higher ROI)

