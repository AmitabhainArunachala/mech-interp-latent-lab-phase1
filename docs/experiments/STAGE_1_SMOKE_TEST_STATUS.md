# Stage 1 Smoke Test — Status

**Date:** January 5, 2025  
**Status:** 🔄 **IN PROGRESS**

---

## Smoke Test Experiments

### Test 1: L0 Necessity (Ablation)
- **Script:** `scripts/smoke_test_l0_necessity.py`
- **Config:** `configs/mlp_ablation_necessity_l0.json` (n_pairs=5 for smoke test)
- **Status:** 🔄 Running
- **Run Directory:** `results/phase1_mechanism/runs/20260105_131922_l0_necessity_smoke_test/`

### Test 2: L0 Sufficiency (Patch)
- **Script:** `scripts/smoke_test_l0_sufficiency.py`
- **Config:** `configs/mlp_sufficiency_l0.json` (n_pairs=5 for smoke test)
- **Status:** ⏳ Pending (waiting for Test 1)

---

## Success Criteria Checklist

### ✅ Infrastructure Ready
- [x] PromptLoader with IDs implemented
- [x] Run metadata helper created
- [x] Metric contract standardized
- [x] Pipelines updated

### ⏳ Smoke Test Validation (Pending)
- [ ] CSV has `recursive_prompt_id` column
- [ ] CSV has `baseline_prompt_id` column
- [ ] `summary.json` has `git_commit` key
- [ ] `summary.json` has `prompt_bank_version` key
- [ ] `summary.json` has `mode_score_m` key
- [ ] `metadata.json` exists in run directory
- [ ] `results/RUN_INDEX.jsonl` exists and updated

---

## Monitoring

**Check status:**
```bash
ssh runpod-current "tail -f /tmp/smoke_test_l0_necessity.log"
```

**Check if running:**
```bash
ssh runpod-current "ps aux | grep smoke_test | grep python"
```

**Verify results (after completion):**
```bash
ssh runpod-current "cd /root/mech-interp-latent-lab-phase1 && find results/phase1_mechanism/runs -name '*smoke_test*' -type d | sort | tail -1 | xargs -I {} cat {}/summary.json | python3 -m json.tool"
```

---

**Next:** Once Test 1 completes and passes all criteria, run Test 2.


