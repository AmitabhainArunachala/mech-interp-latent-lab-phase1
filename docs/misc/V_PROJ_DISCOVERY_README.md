# V-Projection Head Discovery - Simplified Pipeline

**Status:** Ready to run  
**Method:** V-projection ablation (proven to work)

---

## What This Does

Uses **V-projection ablation** - the method that successfully found L27H11/H1/H22 in `HEAD_ABLATION_RESULTS.md`.

**How it works:**
1. Zero out V-projection values for a specific head BEFORE attention computation
2. Measure R_V change
3. Heads with larger |delta| are more important

**Why this works:**
- More reliable than modifying attention weights
- Directly affects the value vectors used in attention
- No sequence length issues (works on any prompt length)

---

## Run It

### On GPU:
```bash
ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44
cd /workspace/mech-interp-latent-lab-phase1
python3 v_proj_head_discovery.py
```

### Or run in background:
```bash
ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 "cd /workspace/mech-interp-latent-lab-phase1 && nohup python3 v_proj_head_discovery.py > v_proj_discovery.log 2>&1 &"
```

---

## What You Get

**CSV Output:** `results/head_discovery/v_proj_head_discovery_YYYYMMDD_HHMMSS.csv`

**Columns:**
- `layer`: Layer index (8-27)
- `head`: Head index (0-31)
- `rv_baseline`: Baseline R_V (no ablation)
- `rv_ablated`: R_V after ablating this head
- `delta`: Change in R_V (positive = increases R_V, negative = decreases R_V)
- `abs_delta`: Absolute delta (for ranking)
- `n_samples`: Number of prompts tested

---

## Interpreting Results

### Important Heads:
- **|delta| > 0.02**: Important head (2%+ change in R_V)
- **delta < 0**: Head causes contraction (decreases R_V when active)
- **delta > 0**: Head prevents contraction (increases R_V when active)

### Known Important Heads (to validate):
- **L27H11:** Should show Δ ≈ +0.06 (6% impact)
- **L27H1:** Should show Δ ≈ +0.03 (3% impact)
- **L27H22:** Should show Δ ≈ +0.02 (2.4% impact)

---

## Expected Runtime

- **20 layers × 32 heads = 640 tests**
- **20 prompts per test**
- **~2-3 seconds per test**
- **Total: ~20-30 minutes**

---

## Advantages Over Previous Approach

1. ✅ **Simple:** Just zero V-projection values
2. ✅ **Reliable:** No sequence length issues
3. ✅ **Proven:** Same method that found L27H11/H1/H22
4. ✅ **Fast:** Direct hook, no complex patching
5. ✅ **Works:** Actually modifies the computation

---

## Next Steps After Discovery

1. **Validate top heads:** Check if L27H11/H1/H22 show up
2. **Find new heads:** Discover heads at other layers
3. **Visualize:** Create attention pattern heatmaps for top heads
4. **Test sufficiency:** Can we reproduce effect with just top heads?
5. **Test necessity:** Does ablating top heads break the effect?

---

**This should work! It's the same method that already worked before.**









