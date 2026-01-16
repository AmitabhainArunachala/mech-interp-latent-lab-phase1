# UNIFIED DIRECTIVE: Gemma 2 9B Circuit Discovery

**Date:** January 16, 2026
**Status:** AUTHORITATIVE - This supersedes conflicting docs
**SSH:** `ssh root@198.13.252.23 -p 12619`

---

## SITUATION SUMMARY

Your agent found doc conflicts and an audit showing "industry-grade not yet achieved." Here's the resolution:

### Doc Conflicts Resolved

| Issue | Resolution |
|-------|------------|
| **Threshold discrepancy** (ENTRYGATE: <0.75 vs MULTI_MODEL: <0.65) | **Use ENTRYGATE thresholds** (<0.75 for champions, >0.85 for controls). These are more conservative and appropriate for cross-architecture. |
| **Phase numbering** (ENTRYGATE: 2-6 vs MULTI_MODEL: 1-7) | **Use ENTRYGATE phases**. Characterization JSON is nice-to-have, not blocking. |
| **Head identification** | **Include it** - run `05_head_identification.json` config |

### Audit Findings: What Matters for THIS Run

The audit (Jan 5, 2026) found infrastructure gaps on *historical runs*. For THIS Gemma run:

| Audit Finding | Impact on Gemma Run | Action |
|---------------|---------------------|--------|
| RUN_INDEX.jsonl missing | Low - we'll create artifacts fresh | Ensure summary.json created |
| mode_score_m not unified | **Medium** - should compute it | Add to Phase 2 output |
| Docs vs disk contradictions | N/A - new run | Create clean artifacts |

**Bottom line:** The audit is about historical runs. Your Gemma run should CREATE proper artifacts, not fix old ones.

---

## AUTHORITATIVE THRESHOLDS (USE THESE)

### Phase 2 Gate Criteria

| Criterion | Threshold | Hard/Soft |
|-----------|-----------|-----------|
| Champions R_V | < 0.75 | **HARD** |
| Controls R_V | > 0.85 | **HARD** |
| Cohen's d | > 1.0 (absolute) | **HARD** |
| p-value | < 0.01 | **HARD** |
| n (each group) | ≥ 40 | **HARD** |
| mode_score_m computed | Yes | **SOFT** (compute but don't gate on it) |

### Phase 3/4 Success Criteria

| Phase | Success | Threshold |
|-------|---------|-----------|
| Phase 3 (Source) | Ablation removes contraction | R_V delta > +0.3 |
| Phase 4 (Transfer) | Steering induces contraction | R_V delta < -0.1 |

---

## EXECUTION SEQUENCE (DEFINITIVE)

### Phase 0: Preflight (5 min)
```bash
ssh root@198.13.252.23 -p 12619
nvidia-smi  # Confirm GPU
cd /workspace/mech-interp-latent-lab-phase1
git pull

# Verify environment
python -c "from prompts.loader import PromptLoader; print(PromptLoader().version)"
python -c "from src.metrics import compute_rv, compute_extended_metrics; print('OK')"
```

### Phase 1: Generate Configs (5 min)
```bash
python scripts/generate_model_configs.py \
  --model google/gemma-2-9b \
  --output-dir configs/discovery

ls configs/discovery/gemma_2_9b/
# Expect 23 files
```

**Gemma 2 9B specs:**
- Layers: 42
- Late layer: 35 (84%)
- Heads: 16
- KV heads: 8 (GQA)

### Phase 2: Baseline R_V (GATE) (1-2 hours)
```bash
python -m src.pipelines.run \
  --config configs/discovery/gemma_2_9b/01_baseline_rv.json
```

**Check results:**
```bash
cat results/phase2_generalization/gemma_2_9b/*/summary.json | python -m json.tool
```

**GATE DECISION:**
- ALL thresholds pass → Continue
- 0.5 < |d| < 1.0 → Continue with caution, note weak effect
- |d| < 0.5 OR wrong direction → **STOP**, document failure

### Phase 3: Source Hunt (6-12 hours)
```bash
for layer in 0 1 2 3 4 5 6 7 8; do
  echo "=== Ablating L$layer ==="
  python -m src.pipelines.run \
    --config configs/discovery/gemma_2_9b/02_source_hunt_mlp_ablation_l${layer}.json
done
```

### Phase 4: Transfer Hunt (7-14 hours)
```bash
for layer in 0 1 2 3 4 5 6 7 8 9 10; do
  echo "=== Steering L$layer ==="
  python -m src.pipelines.run \
    --config configs/discovery/gemma_2_9b/03_transfer_hunt_mlp_steer_l${layer}.json
done
```

**IF STEERING CAUSES EXPANSION:**
- This happened on Llama - it's a real finding, not a bug
- Document it
- Note: May need V-projection steering instead of residual stream

### Phase 5: Readout Validation (1-2 hours)
```bash
python -m src.pipelines.run \
  --config configs/discovery/gemma_2_9b/04_readout_validation.json
```

### Phase 6: Head Identification (2-4 hours)
```bash
python -m src.pipelines.run \
  --config configs/discovery/gemma_2_9b/05_head_identification.json
```

### Phase 7: Extended Metrics (1 hour)
```python
from src.metrics import compute_extended_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

# Compute on sample prompts
from prompts.loader import PromptLoader
loader = PromptLoader()

champions = loader.get_group("champions")[:5]
controls = loader.get_group("baseline_math")[:5]

results = {"champions": [], "controls": []}

for p in champions:
    ext = compute_extended_metrics(model, tokenizer, p["text"], early_layer=5, late_layer=35)
    results["champions"].append(ext.to_dict())

for p in controls:
    ext = compute_extended_metrics(model, tokenizer, p["text"], early_layer=5, late_layer=35)
    results["controls"].append(ext.to_dict())

import json
with open("results/phase2_generalization/gemma_2_9b/extended_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## OUTPUT REQUIREMENTS

### Required Artifact: CIRCUIT_MAP.md

Create `results/phase2_generalization/gemma_2_9b/CIRCUIT_MAP.md`:

```markdown
# Gemma 2 9B R_V Circuit Map

**Date:** January 16, 2026
**Model:** google/gemma-2-9b
**Status:** [VALIDATED / PARTIAL / NOT FOUND]

## Model Architecture
- Layers: 42
- Heads: 16
- KV heads: 8 (GQA)
- Hidden dim: 3584
- Late layer: 35 (84% depth)

## Phase 2: Baseline R_V Separation

| Group | R_V Mean | Std | n | 95% CI |
|-------|----------|-----|---|--------|
| Champions | X.XX | X.XX | XX | [X.XX, X.XX] |
| Controls | X.XX | X.XX | XX | [X.XX, X.XX] |

**Effect size:** d = X.XX
**p-value:** X.XXe-XX
**Gate:** [PASS / WEAK PASS / FAIL]

## Phase 3: Source Layer (MLP Ablation)

| Layer | R_V Baseline | R_V Ablated | Delta | p-value | Role |
|-------|--------------|-------------|-------|---------|------|
| L0 | X.XX | X.XX | +X.XX | X.XX | [PRIMARY/SECONDARY/NONE] |
| L1 | X.XX | X.XX | +X.XX | X.XX | |
| ... | | | | | |

**Source Layer:** L[X]

## Phase 4: Transfer Layer (MLP Steering)

| Layer | R_V Baseline | R_V Steered | Delta | p-value | Effect |
|-------|--------------|-------------|-------|---------|--------|
| L0 | X.XX | X.XX | X.XX | X.XX | [TRANSFER/NONE/EXPANSION] |
| L1 | X.XX | X.XX | X.XX | X.XX | |
| ... | | | | | |

**Transfer Layer:** L[X] (or NOT FOUND - document anomaly)

## Phase 5: Readout Validation (Four-Way Control)

| Condition | Transfer % | p-value | Expected |
|-----------|------------|---------|----------|
| Target (L35→L35) | XX% | X.XX | >50% |
| Wrong layer (L29→L35) | XX% | X.XX | <20% |
| Random patch | XX% | X.XX | <20% |
| Shuffled patch | XX% | X.XX | <20% |

**Readout validated:** [YES / NO]

## Phase 6: Head Identification

| Head | Contribution | p-value | Critical? |
|------|--------------|---------|-----------|
| H[X] | X.XX | X.XX | [YES/NO] |
| ... | | | |

## Phase 7: Extended Metrics

| Metric | Champions | Controls | Delta |
|--------|-----------|----------|-------|
| cosine_early_late | X.XX | X.XX | X.XX |
| spectral_late_top1_ratio | X.XX | X.XX | X.XX |
| spectral_late_spectral_gap | X.XX | X.XX | X.XX |
| attention_entropy | X.XX | X.XX | X.XX |

## Circuit Comparison to Reference

| Component | Mistral-7B | Llama-3-8B | Gemma-2-9B | Notes |
|-----------|------------|------------|------------|-------|
| Source | L0 MLP | L0 MLP | L[?] | |
| Transfer | L3-L4 MLP | NOT FOUND | L[?] | |
| Readout | L27 | L27 | L35 | |
| Effect (d) | -3.56 | -1.34 | [?] | |

## Anomalies / Differences

[Document any unexpected findings here]

## Conclusions

[Summary paragraph]

## Raw Data Paths

- Phase 2: `results/phase2_generalization/gemma_2_9b/[TIMESTAMP]_cross_architecture_validation/`
- Phase 3: `results/phase2_generalization/gemma_2_9b/[TIMESTAMP]_mlp_ablation_necessity/`
- Phase 4: `results/phase2_generalization/gemma_2_9b/[TIMESTAMP]_combined_mlp_sufficiency/`
- Phase 5: `results/phase2_generalization/gemma_2_9b/[TIMESTAMP]_rv_l27_causal_validation/`
- Phase 6: `results/phase2_generalization/gemma_2_9b/[TIMESTAMP]_head_ablation_validation/`
- Extended: `results/phase2_generalization/gemma_2_9b/extended_metrics.json`
```

---

## WHAT NOT TO DO

| Don't | Why |
|-------|-----|
| Change prompt bank | Breaks comparability |
| Skip Phase 2 gate check | Wasted compute |
| Proceed with n < 30 | Underpowered |
| Ignore steering anomalies | Real findings |
| Try to fix historical runs | Out of scope |
| Use MULTI_MODEL thresholds | Use ENTRYGATE |

---

## TIME ESTIMATE

| Phase | Time |
|-------|------|
| 0-1 (setup) | 15 min |
| 2 (gate) | 1-2 hr |
| 3 (ablation ×9) | 6-12 hr |
| 4 (steering ×11) | 7-14 hr |
| 5 (validation) | 1-2 hr |
| 6 (heads) | 2-4 hr |
| 7 (extended) | 1 hr |
| **Total** | **18-36 hours** |

---

## SUCCESS CRITERIA

The run is successful if:

1. ✅ Phase 2 gate passes (or documents failure)
2. ✅ Source layer identified (or documented as different)
3. ✅ Transfer layer identified (or anomaly documented)
4. ✅ Readout validated with 4-way controls
5. ✅ Extended metrics computed
6. ✅ CIRCUIT_MAP.md created with all tables filled
7. ✅ All raw data saved to `results/phase2_generalization/gemma_2_9b/`

---

**This document is authoritative. Follow it exactly.**
