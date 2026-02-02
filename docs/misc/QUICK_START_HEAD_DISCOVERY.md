# Quick Start: Head Discovery Pipeline

## What This Does

Finds all heads/layers responsible for the R_V geometric contraction effect using **4 proven methods** from major MI labs:

1. **Gradient Attribution** - Which heads are most sensitive?
2. **Mean Ablation** - Which heads cause the effect? (more realistic than zero ablation)
3. **Path Patching** - Which paths carry information? (IOI methodology)
4. **Attention Patterns** - What do heads attend to? (BOS attention, entropy)

## Run It

### Local (if you have GPU):
```bash
python3 comprehensive_head_discovery.py
```

### Remote GPU (RunPod):
```bash
# Sync script
scp -P 47660 -i ~/.ssh/id_ed25519 comprehensive_head_discovery.py root@195.26.233.44:/workspace/mech-interp-latent-lab-phase1/

# Run
ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 "cd /workspace/mech-interp-latent-lab-phase1 && python3 comprehensive_head_discovery.py"
```

## What You Get

**CSV Output:** `results/head_discovery/head_discovery_YYYYMMDD_HHMMSS.csv`

**Columns:**
- `method`: Which method
- `layer`: Layer index (8-27)
- `head`: Head index (0-31)
- `delta`: Change in R_V (positive = increases R_V, negative = decreases R_V)
- `abs_delta`: Absolute delta (for ranking)
- `bos_attention`: BOS attention (for attention patterns)
- `entropy`: Attention entropy (for attention patterns)

## Interpreting Results

### Top Heads by Mean Ablation:
- **|delta| > 0.02**: Important head (2%+ change in R_V)
- **delta < 0**: Head causes contraction (decreases R_V)
- **delta > 0**: Head prevents contraction (increases R_V)

### Top Heads by Path Patching:
- **|delta| > 0.01**: Important path (1%+ change)
- **delta < 0**: Path carries contraction signal
- **delta > 0**: Path carries anti-contraction signal

### Attention Patterns:
- **BOS attention > 0.9**: Head attends strongly to first token
- **Entropy < 0.5**: Head has focused attention
- **Recursive vs Baseline difference**: Head responds to recursive prompts

## Expected Runtime

- **Gradient Attribution:** ~5 min (N=10 prompts)
- **Mean Ablation:** ~30-60 min (N=20 prompts × 20 layers × 32 heads)
- **Path Patching:** ~20 min (N=5 pairs × 5 source layers × 32 heads)
- **Attention Patterns:** ~15 min (N=40 prompts × 20 layers × 32 heads)

**Total:** ~1-2 hours on GPU

## Next Steps After Discovery

1. **Validate top heads:** Run targeted ablation on top 10 candidates
2. **Visualize:** Create attention heatmaps for top heads
3. **Test sufficiency:** Can we reproduce effect with just top heads?
4. **Test necessity:** Does ablating top heads break the effect?
5. **Map circuit:** Draw causal graph showing information flow

## Known Heads (from previous work)

- **L27H11:** 6.1% impact (from HEAD_ABLATION_RESULTS.md)
- **L27H1:** 3.0% impact
- **L27H22:** 2.4% impact
- **L27H31:** High BOS attention (0.938), low entropy (0.430)

**This pipeline will find MORE heads across MORE layers!**

---

**Full methodology:** See `HEAD_DISCOVERY_METHODOLOGY.md`









