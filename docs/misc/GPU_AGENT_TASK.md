# GPU Agent Task: Run Gold Standard Suite

**Date:** December 16, 2025  
**Estimated time:** 30-45 minutes

---

## Quick Onboarding (5 min)

1. Read: `QUICK_START.md`  
2. Read: `GOLD_STANDARD_SUITE.md`

---

## GPU Setup

```bash
# SSH to GPU
ssh -p 26018 -i ~/.ssh/id_ed25519 root@82.221.170.234

# Navigate and set cache
cd /workspace/mech-interp-latent-lab-phase1
export HF_HOME=/workspace/.hf
export HUGGINGFACE_HUB_CACHE=/workspace/.hf/hub
```

---

## Run These Pipelines (in order)

### Pipeline 1: Existence
Does R_V contraction exist with confound controls?

```bash
python3 -m src.pipelines.run --config configs/gold/01_existence.json
```

### Pipeline 2: Causality  
Is L27 causal for geometric contraction?

```bash
python3 -m src.pipelines.run --config configs/gold/02_causality.json
```

### Pipeline 4: Head Validation
Do KV-head groups drive contraction?

```bash
python3 validate_h18_h26_gold_standard.py
```

---

## After Running

### Check Results
```bash
# View summaries
cat results/gold_standard/runs/*/summary.json
cat results/h18_h26_gold_standard/*.json
```

### Sync Results Back to Local
```bash
# Run this from LOCAL machine (not GPU)
scp -r -P 26018 -i ~/.ssh/id_ed25519 root@82.221.170.234:/workspace/mech-interp-latent-lab-phase1/results/ /Users/dhyana/mech-interp-latent-lab-phase1/results/
```

---

## Pass Criteria

### Pipeline 1 (Existence)
- [ ] Champions R_V < 0.6
- [ ] Champions < length_matched (p < 0.001)
- [ ] Champions < pseudo_recursive (p < 0.001)

### Pipeline 2 (Causality)
- [ ] Transfer efficiency > 50%
- [ ] Patched < baseline (p < 0.001)
- [ ] Wrong-layer control null (p > 0.05)

### Pipeline 4 (Head Validation)
- [ ] Target ablation increases R_V (p < 0.001)
- [ ] Target > control head (p < 0.01)
- [ ] L27 > L21 (p < 0.001)

---

## Key Constraints

- Use ONLY `src/metrics/rv.py` for R_V calculations
- Use `prompts/loader.py` to access prompts
- Model: `mistralai/Mistral-7B-v0.1`
- Log `PromptLoader().version` hash in results

---

## Context

See `agent_reviews/responses/` for 5 independent audits of repo status.

**Key finding:** Geometry claims are solid (95%), behavior claims are fragile (40%).

---

## Troubleshooting

**If disk full:**
```bash
export HF_HOME=/workspace/.hf
```

**If module not found:**
```bash
cd /workspace/mech-interp-latent-lab-phase1
```

**If prompts not loading:**
```bash
# Check prompt bank exists
ls -la prompts/bank.json
```









