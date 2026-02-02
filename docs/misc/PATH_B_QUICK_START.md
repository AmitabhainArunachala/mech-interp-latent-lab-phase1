# Path B Validation: Quick Start Guide

**Status:** ✅ Ready to run  
**Total Runtime:** ~10-13 hours GPU time

---

## 🚀 Quick Start

### Option 1: Run All Experiments Sequentially

```bash
# On RunPod GPU
ssh -p 53317 -i ~/.ssh/id_ed25519 root@195.26.233.61
cd /workspace/mech-interp-latent-lab-phase1

# Run in sequence (each will take 2-6 hours)
python3 experiment_multi_token_generation.py
python3 experiment_kv_only_control.py
python3 experiment_hysteresis.py
```

### Option 2: Run in Parallel (if you have multiple GPUs)

```bash
# Terminal 1
python3 experiment_multi_token_generation.py > multi_token.log 2>&1 &

# Terminal 2
python3 experiment_kv_only_control.py > kv_only.log 2>&1 &

# Terminal 3
python3 experiment_hysteresis.py > hysteresis.log 2>&1 &

# Monitor progress
tail -f multi_token.log
```

---

## 📊 Expected Outputs

### Experiment 1: Multi-Token Generation
- `results/path_b_validation/runs/TIMESTAMP_multi_token_generation/`
  - `all_trajectories.csv` - All step-by-step data
  - `persistence_summary.csv` - Aggregated metrics
  - `summary.json` - Statistical summary

### Experiment 2: KV-Only Control
- `results/path_b_validation/runs/TIMESTAMP_kv_only_control/`
  - `results.csv` - All condition results
  - `summary.json` - Statistical summary

### Experiment 3: Hysteresis
- `results/path_b_validation/runs/TIMESTAMP_hysteresis/`
  - `results.csv` - All pair results
  - `summary.json` - Statistical summary by layer

---

## ✅ Success Criteria

### Experiment 1 (Multi-Token)
- ✅ Recursive: Persistence ratio > 0.7 (maintains contraction)
- ✅ Baseline: Persistence ratio < 0.3 (no contraction)
- ✅ Clear separation in trajectories

### Experiment 2 (KV-Only)
- ✅ KV-only: Expression rate > 20% (vs control < 5%)
- ✅ KV+V_PROJ: Expression rate > KV-only
- ✅ Random KV: Expression rate ≈ control

### Experiment 3 (Hysteresis)
- ✅ Forward recovery > 80%
- ✅ Reverse recovery < 20%
- ✅ Asymmetry > 50%, p < 0.05

---

## 🔍 What to Check After Running

1. **Check logs for errors** - Look for exceptions or NaN values
2. **Verify sample sizes** - Ensure all N pairs completed
3. **Check summary.json** - Review statistical summaries
4. **Compare to expectations** - Do results match success criteria?

---

## 📝 Next Steps After Results

1. **Analyze results** - Check if experiments validate/falsify claims
2. **Update STRATEGIC_ROADMAP** - Revise based on findings
3. **Update THE_BIG_QUESTIONS** - Mark resolved questions
4. **Plan follow-ups** - If gaps remain, design targeted experiments

---

## 🐛 Troubleshooting

### Out of Memory
- Reduce `N_PAIRS` or `MAX_GENERATION_STEPS`
- Clear GPU cache between runs: `torch.cuda.empty_cache()`

### Slow Performance
- Check GPU utilization: `nvidia-smi`
- Ensure `attn_implementation="eager"` is set

### Missing Prompts
- Check `prompts/bank.json` exists
- Fallback to `REUSABLE_PROMPT_BANK` should work

---

**Ready to run! Start with Experiment 1 (multi-token generation).**









