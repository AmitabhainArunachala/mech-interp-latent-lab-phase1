# Session Summary: January 15, 2026

## GPU Session: Llama-3-8B R_V Circuit Discovery

**Duration:** ~2.5 hours  
**Server:** RunPod (216.81.151.42:18748)  
**Model Tested:** meta-llama/Meta-Llama-3-8B (base)

---

## What We Did

1. **Ran cross-architecture validation** on Llama-3-8B base model
2. **Discovered source layer** through MLP ablation sweep (L0-L8)
3. **Tested transfer mechanism** through residual steering sweep (L0-L15)
4. **Created CIRCUIT_MAP.md** documenting findings

---

## Key Results

### ✅ R_V Contraction Present on Llama
| Prompt Type | R_V | Interpretation |
|-------------|-----|----------------|
| Champions | 0.72 ± 0.10 | **Contraction** ✓ |
| L4_full | 0.72 ± 0.03 | **Contraction** ✓ |
| Length-matched | 0.95 ± 0.21 | No contraction |
| Pseudo-recursive | 0.78 ± 0.10 | Also contracts (anomaly!) |

### ✅ Source Layer: L0 MLP (Same as Mistral!)
| Layer | R_V after ablation | Effect |
|-------|-------------------|--------|
| L0 | 2.19 | +1.44 (removes effect) |
| L1 | 1.89 | +1.14 |
| L5-L8 | ~0.75 | No change |

### ❌ Transfer Mechanism: Different from Mistral
- Steering at L0-L4 causes **expansion** (opposite direction!)
- No layer found where steering increases contraction
- Mistral's L3-L4 transfer doesn't replicate on Llama

---

## Cross-Architecture Comparison

| Component | Mistral-7B | Llama-3-8B | Match? |
|-----------|------------|------------|--------|
| R_V contraction | ✓ Present | ✓ Present | ✅ |
| Source layer | L0 MLP | L0 MLP | ✅ |
| Transfer layer | L3-L4 | Not found | ❌ |
| Effect strength | d = -3.56 | d = -1.34 | Weaker |
| Pseudo-recursive | Different | Same as champions | Anomaly |

---

## Files Saved Locally

```
results/phase2_generalization/
├── llama3_8b_base/
│   ├── CIRCUIT_MAP.md              # Full circuit documentation
│   ├── source_layer_sweep.log      # Raw ablation results
│   ├── transfer_layer_sweep.log    # Raw steering results
│   └── 20260115_171531_cross_architecture_validation/
│       ├── summary.json
│       └── cross_architecture_validation.csv
└── JAN15_2026_SESSION_SUMMARY.md   # This file
```

---

## Implications

1. **R_V contraction is cross-architecture** - Not Mistral-specific
2. **Source circuit is shared** - L0 MLP is universal
3. **Transfer mechanism is architecture-specific** - May need different intervention for Llama
4. **Llama is less selective** - Pseudo-recursive triggers same response as champions

---

## Next Steps (When Time Permits)

- [ ] Try V-projection head steering on Llama (H18+H26 equivalent)
- [ ] Test on Mixtral for MoE amplification
- [ ] Investigate pseudo-recursive anomaly
- [ ] Compare Llama-3-8B base vs instruct

---

**Status:** ✅ All results synced to local repo. Safe to close RunPod.
