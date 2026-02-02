# Head Discovery Pipeline Status

**Started:** Running now on RunPod GPU  
**Process ID:** Active  
**GPU:** NVIDIA RTX 6000 Ada Generation

---

## Progress

### ✅ Completed
- Model loaded (Mistral-7B-v0.1)
- Prompts loaded (30 recursive, 25 baseline)
- Pipeline initialized

### 🔄 In Progress
**Step 1/4: Gradient Attribution Analysis**
- Progress: 2/10 prompts (~20%)
- Speed: ~20 seconds per prompt
- ETA: ~3-4 minutes remaining

### ⏳ Pending
**Step 2/4: Mean Ablation Analysis**
- Will test: 20 layers × 32 heads = 640 combinations
- Sample size: 20 prompts per head
- Estimated time: 30-60 minutes

**Step 3/4: Path Patching Analysis**
- Will test: 5 source layers × 32 heads = 160 combinations
- Sample size: 5 prompt pairs
- Estimated time: 20 minutes

**Step 4/4: Attention Pattern Analysis**
- Will test: 20 layers × 32 heads = 640 combinations
- Sample size: 40 prompts (20 recursive + 20 baseline)
- Estimated time: 15 minutes

---

## Total Estimated Runtime

- **Gradient Attribution:** ~3-4 min (in progress)
- **Mean Ablation:** ~30-60 min
- **Path Patching:** ~20 min
- **Attention Patterns:** ~15 min

**Total:** ~1-2 hours

---

## Monitor Progress

**Quick check:**
```bash
ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 "cd /workspace/mech-interp-latent-lab-phase1 && tail -20 head_discovery_run.log"
```

**Check results:**
```bash
ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 "cd /workspace/mech-interp-latent-lab-phase1 && ls -lh results/head_discovery/*.csv 2>/dev/null || echo 'No CSV yet'"
```

**Use monitoring script:**
```bash
./monitor_head_discovery.sh
```

---

## What We're Looking For

### Top Heads (by Mean Ablation):
- **|delta| > 0.02**: Important head (2%+ change in R_V)
- **delta < 0**: Head causes contraction (decreases R_V)
- **delta > 0**: Head prevents contraction (increases R_V)

### Known Important Heads (to validate):
- **L27H11:** 6.1% impact (should show up!)
- **L27H1:** 3.0% impact
- **L27H22:** 2.4% impact
- **L27H31:** High BOS attention (0.938)

### Expected Discoveries:
- More heads at L27 (beyond H11/H1/H22/H31)
- Important heads at earlier layers (8-26)
- Paths from early → late layers
- Attention patterns similar to H31

---

**Pipeline is running! Check back in ~1-2 hours for full results.**









