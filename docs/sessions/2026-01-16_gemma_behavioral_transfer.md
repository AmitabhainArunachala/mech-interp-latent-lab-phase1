# Gemma 2 9B Behavioral Transfer Experiments

**Date**: 2026-01-16
**Goal**: Test if R_V geometric interventions produce behavioral domain shifts on Gemma 2 9B

## Summary

**Key Finding**: Gemma 2 9B shows **geometry transfer without behavior transfer**. V-proj steering produces strong R_V contraction (32% reduction) but **zero behavioral domain shift**.

This contrasts sharply with Mistral 7B, where the same interventions produce 15-20% domain shifts.

## Experiments Run

### Experiment 09: V-proj Patching (PersistentVPatcher)
- **Method**: Extract V_proj from recursive prompt, patch into baseline during generation at L35
- **Result**: 0% domain shift, 0% philosophical markers
- **R_V Transfer Delta**: -0.049 (slight contraction)
- **Baseline domain**: 37% narrative, 37% task, 20% other
- **Transfer domain**: IDENTICAL to baseline

### Experiment 10: C2 KV Full Cascade
- **Method**: Full KV cache swap + residual steering
- **Result**: FAILED - GQA architecture incompatible
- **Error**: "can only concatenate tuple (not 'Tensor') to tuple"
- **Root cause**: Gemma's GQA KV cache structure differs from Mistral's MHA

### Experiment 11: V-proj Steering Only (L35, alpha=2.5)
- **Method**: Add steering vector to V_proj output during generation
- **Result**:
  - Baseline: R_V=0.996, 0% philosophical, 35% task
  - Steered: R_V=0.677, 0% philosophical, 35% task
- **Key observation**: **32% R_V contraction but identical behavior**

### Experiment 12: Mid-layer High Alpha (L20, alpha=5.0)
- **Method**: Steer earlier in network with stronger alpha
- **Result**:
  - Baseline: R_V=0.996, 0% philosophical, 35% task
  - Steered: R_V=0.990, 0% philosophical, 35% task
- **Key observation**: Mid-layer steering doesn't contract R_V as much AND doesn't change behavior

## Cross-Architecture Comparison

| Experiment | Mistral 7B (L27) | Gemma 2 9B (L35) |
|------------|------------------|------------------|
| V-proj patching | ~15-20% domain shift | 0% domain shift |
| V-proj steering | Strong behavioral effect | R_V contracts, 0% behavior change |
| C2 full (KV swap) | 0% → 20% philosophical | GQA incompatible |

## Root Cause Analysis

Deployed two subagents to diagnose. Key findings:

### 1. GQA Bandwidth Limitation
- Mistral: 4:1 ratio (8 KV heads → 32 Q heads) = concentrated signal
- Gemma: 2:1 ratio (8 KV heads → 16 Q heads) = dispersed signal
- **50% less steering bandwidth in Gemma**

### 2. Deeper Architecture Allows Recovery
- Mistral: L27 with 5 downstream layers (16% of network)
- Gemma: L35 with 7 downstream layers (17% of network)
- **More layers = more opportunity for steering signal to decay**

### 3. Local+Global Attention Windows
- Gemma uses alternating local (sliding window) and global attention
- Local attention can "reset" steering signal when context moves outside window
- Mistral uses full-sequence attention throughout

### 4. Geometry ≠ Behavior
- R_V measures point-in-time geometry (single forward pass)
- Behavior measures sustained generation across 100+ tokens
- Steering may change geometry instantaneously but decay during generation

## Implications for Research

1. **The geometry→behavior link is architecture-dependent**, not universal
2. **GQA models may be more robust to activation interventions** - this could be a safety feature or a limitation
3. **R_V contraction is necessary but not sufficient** for behavioral transfer
4. **KV cache interventions need architecture-specific implementations** for GQA models

## Open Questions

1. Would multi-layer steering (L20 + L35 simultaneously) produce behavioral effects?
2. Is there a GQA-compatible KV patching approach?
3. Does Gemma's robustness extend to other intervention types (SAE steering, CAA)?
4. Is this a Gemma-specific phenomenon or common to all GQA models?

## Files Modified

- `src/pipelines/discovery/c2_rv_measurement.py` - Added configurable `steering_layer` parameter
- `src/pipelines/discovery/vproj_patching_analysis.py` - Added configurable R_V layers
- Created configs: `09_vproj_patching_behavioral.json`, `10_c2_behavioral_kv_full.json`, `11_cascade_residual_only.json`, `12_midlayer_high_alpha.json`

## Next Steps (Future Session)

1. Try persistent multi-layer patching (L3 source + L35 peak)
2. Investigate GQA-compatible KV intervention approaches
3. Test on Qwen2 7B (also GQA) to see if pattern holds
4. Consider residual stream interventions instead of attention interventions
