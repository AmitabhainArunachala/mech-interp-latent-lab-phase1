# GPU Status: Ready and Running

## ✅ Setup Complete

**GPU:** NVIDIA RTX 6000 Ada Generation  
**VRAM:** 49 GB (48.5 GB free)  
**CUDA:** Available  
**PyTorch:** 2.4.1+cu124  

**SSH Command:**
```bash
ssh root@195.26.233.44 -p 47660 -i ~/.ssh/id_ed25519
```

**Remote Directory:**
```
/workspace/mech-interp-latent-lab-phase1
```

## 🚀 Currently Running

**H31 Validation Script:** `h31_validation_n50.py`
- **Status:** Running (process 1157)
- **Task:** Analyzing 50 recursive + 50 baseline prompts
- **Measures:** H31 BOS attention, entropy, R_V at L27
- **Output:** `results/h31_validation/h31_validation_n50.csv`

## 📊 Monitor Progress

**Quick check:**
```bash
ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 "cd /workspace/mech-interp-latent-lab-phase1 && tail -20 h31_validation_run.log"
```

**Check CSV:**
```bash
ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 "cd /workspace/mech-interp-latent-lab-phase1 && wc -l results/h31_validation/h31_validation_n50.csv 2>/dev/null || echo 'Not ready yet'"
```

**Download results when done:**
```bash
scp -P 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44:/workspace/mech-interp-latent-lab-phase1/results/h31_validation/h31_validation_n50.csv ./results/h31_validation/
```

## ⏱️ Estimated Runtime

- **Model loading:** ~1-2 minutes
- **50 prompts:** ~10-15 minutes total
- **Expected completion:** ~15-20 minutes from start

## 🎯 What We're Validating

**Hypothesis:** H31 entropy perfectly separates recursive (0.28) vs baseline (0.81)

**Test:** 50 recursive + 50 baseline prompts

**Success criteria:**
- Recursive mean entropy < 0.4
- Baseline mean entropy > 0.65  
- Clear separation (no overlap)
- Statistical significance

---

**GPU is online and running! Check progress with the commands above.**









