# Head Discovery Pipeline - Rerun Status

**Started:** Running now with sequence length fixes  
**Log file:** `head_discovery_run_v2.log`  
**GPU:** NVIDIA RTX 6000 Ada Generation

---

## 🔧 Fixes Applied

1. **Mean Ablation:** Now handles variable sequence lengths with interpolation/padding
2. **Path Patching:** Fixed sequence length matching with truncation/padding
3. **Gradient Attribution:** Should save results properly to CSV

---

## 📊 Current Progress

**Step 1/4: Gradient Attribution Analysis**
- Status: Running
- Progress: Starting (0/10 prompts)
- Speed: ~20 seconds per prompt
- ETA: ~3-4 minutes

**Steps 2-4:** Pending

---

## ⏱️ Estimated Timeline

- **Gradient Attribution:** ~3-4 min (in progress)
- **Mean Ablation:** ~30-60 min
- **Path Patching:** ~20 min  
- **Attention Patterns:** ~15 min

**Total:** ~1-2 hours

---

## 🎯 What We're Testing

1. **Gradient Attribution:** Sensitivity of all heads (10 prompts × 20 layers × 32 heads)
2. **Mean Ablation:** Which heads cause the effect (20 prompts × 20 layers × 32 heads)
3. **Path Patching:** Causal paths between layers (5 pairs × 5 source layers × 32 heads)
4. **Attention Patterns:** What heads attend to (40 prompts × 20 layers × 32 heads)

---

## 📁 Output

Results will be saved to: `results/head_discovery/head_discovery_YYYYMMDD_HHMMSS.csv`

---

## 🔍 Monitor Progress

**Quick check:**
```bash
ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 "cd /workspace/mech-interp-latent-lab-phase1 && tail -20 head_discovery_run_v2.log"
```

**Check for errors:**
```bash
ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 "cd /workspace/mech-interp-latent-lab-phase1 && grep -i error head_discovery_run_v2.log | tail -10"
```

**Check results:**
```bash
ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 "cd /workspace/mech-interp-latent-lab-phase1 && ls -lh results/head_discovery/*.csv | tail -1"
```

---

**Pipeline is running! The sequence length fixes should prevent the NaN errors we saw before.**









