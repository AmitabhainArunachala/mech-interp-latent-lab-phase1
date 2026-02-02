# Reviewer Task: Run Gold Standard Suite

**Goal:** Independently verify all 3 pipelines pass. Log everything.

---

## SSH Access

```bash
ssh -o StrictHostKeyChecking=no -p 26212 -i ~/.ssh/id_ed25519 root@213.173.111.30
cd /workspace/mech-interp-latent-lab-phase1
```

---

## Run These (in order)

```bash
# Pipeline 1: Existence (~1 min)
HF_HOME=/workspace/.hf PYTHONPATH=. python3 -m src.pipelines.run --config configs/gold/01_existence.json

# Pipeline 2: Causality (~2 min)
HF_HOME=/workspace/.hf PYTHONPATH=. python3 -m src.pipelines.run --config configs/gold/02_causality.json

# Pipeline 4: Head Validation (~3 min)
HF_HOME=/workspace/.hf PYTHONPATH=. python3 -m src.pipelines.run --config configs/gold/04_head_validation.json
```

---

## After Each Run, Check

```bash
# Find latest run
ls -lt results/gold_standard/runs/ | head -3

# Check key results (replace TIMESTAMP)
cat results/gold_standard/runs/TIMESTAMP/prompt_bank_version.txt
cat results/gold_standard/runs/TIMESTAMP/summary.json | python3 -m json.tool | head -30

# For Pipeline 4, also check:
cat results/gold_standard/runs/TIMESTAMP/VERDICT.md
```

---

## Pass Criteria

| Pipeline | Must Have |
|----------|-----------|
| 1 | `mean_rv.champions < 0.6`, p < 0.001, `prompt_bank_version.txt` exists |
| 2 | `transfer_percent_estimate` > 50%, wrong-layer opposite direction |
| 4 | `VERDICT.md` shows all ✅, `prompt_bank_version.txt` exists |

---

## Log Format

Please report:
1. **Timestamp** of each run
2. **Key numbers** from summary.json
3. **Any errors** (full traceback)
4. **Pass/Fail** per pipeline

You can log during the run or after. Tag me when done or if issues arise.

---

*Code synced and verified 2025-12-16 06:50 UTC by Opus*









