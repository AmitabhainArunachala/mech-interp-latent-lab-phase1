# MISSION BRIEF: Gemma 2 9B R_V Circuit Discovery

**Date:** January 16, 2026
**Agent:** GPU Runner
**Priority:** HIGH
**SSH Target:** `ssh root@198.13.252.23 -p 12619`

---

## EXECUTIVE SUMMARY

You are running the R_V cross-architecture circuit discovery protocol on **Google's Gemma 2 9B**. This is the industry-standard frontier model for mechanistic interpretability research (Neel Nanda recommends it, Google released Gemma Scope SAEs for it).

**Goal:** Determine if the R_V contraction circuit (validated on Mistral-7B with d = -3.56) generalizes to Gemma 2's architecture.

---

## CONTEXT FROM YESTERDAY

### What We Built
1. **Extended metrics suite** - Added cosine similarity, spectral stats, attention entropy
2. **Merged config generator** - GPT5.2 + Gemini best features combined
3. **Gold-standard protocol** - `docs/CROSS_ARCHITECTURE_DISCOVERY_PIPELINE_ENTRYGATE.md`
4. **Step-by-step plan** - `docs/CROSS_ARCHITECTURE_STEP_BY_STEP_PLAN.md`

### What We Learned from Llama-3-8B Run
- R_V contraction IS present (d = -1.34, weaker than Mistral)
- Source layer L0 MLP confirmed (same as Mistral)
- **Anomaly:** Steering caused EXPANSION not contraction
- **Anomaly:** Pseudo-recursive prompts also contract (unlike Mistral)
- Sample sizes were LOW (n=15 instead of n=50) - watch for this

### Reference Circuit (Mistral-7B Validated)
```
L0 MLP (source) → L3-L4 MLP (transfer) → L27 Attention H18+H26 (readout)
                                              ↓
                                         R_V < 1.0

Key metrics:
- Champions R_V: 0.52 ± 0.05
- Controls R_V: 1.01 ± 0.06
- Cohen's d: -3.56
- p-value: < 10⁻⁶
```

---

## GPU CONNECTION

```bash
ssh root@198.13.252.23 -p 12619
```

---

## PREFLIGHT CHECKLIST

### 1. Verify GPU
```bash
nvidia-smi
# Need: ≥24GB VRAM (Gemma 2 9B needs ~20GB in float16)
```

### 2. Clone/Update Repo
```bash
cd /workspace  # or wherever
git clone https://github.com/[YOUR_REPO]/mech-interp-latent-lab-phase1.git
# OR if exists:
cd mech-interp-latent-lab-phase1 && git pull
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install transformers accelerate bitsandbytes scipy
```

### 4. Verify Environment
```bash
python -c "from prompts.loader import PromptLoader; l = PromptLoader(); print(f'Prompt bank version: {l.version}')"
python -c "from src.metrics import compute_rv, compute_extended_metrics; print('Metrics OK')"
python -c "from src.utils import generate_discovery_configs; print('Config generator OK')"
```

### 5. HuggingFace Login (if needed)
```bash
huggingface-cli login
# Gemma 2 requires accepting license at https://huggingface.co/google/gemma-2-9b
```

---

## EXECUTION PROTOCOL

### Phase 1: Generate Configs

```bash
python scripts/generate_model_configs.py \
  --model google/gemma-2-9b \
  --output-dir configs/discovery

# Verify
ls configs/discovery/gemma_2_9b/
# Should see 23 JSON files
```

**Gemma 2 9B Architecture:**
- Layers: 42
- Heads: 16
- Hidden dim: 3584
- Late layer: 35 (84% of 42)
- GQA: Yes (num_kv_heads = 8)

### Phase 2: Baseline R_V (GATE PHASE)

```bash
python -m src.pipelines.run \
  --config configs/discovery/gemma_2_9b/01_baseline_rv.json
```

**Success Criteria (ALL must pass):**
| Criterion | Threshold |
|-----------|-----------|
| Champions R_V | < 0.75 |
| Controls R_V | > 0.85 |
| Cohen's d | > 1.0 (absolute) |
| p-value | < 0.01 |
| n (each group) | ≥ 40 |

**Check results:**
```bash
cat results/phase2_generalization/gemma_2_9b/*/summary.json | python -m json.tool
```

**DECISION GATE:**
- ALL pass → Continue to Phase 3
- Weak (0.5 < d < 1.0) → Continue with caution
- Fail (d < 0.5) → STOP, document as "does not generalize"

### Phase 3: Source Layer Hunt (MLP Ablation)

```bash
for layer in 0 1 2 3 4 5 6 7 8; do
  echo "=== Ablating Layer $layer ==="
  python -m src.pipelines.run \
    --config configs/discovery/gemma_2_9b/02_source_hunt_mlp_ablation_l${layer}.json
done
```

**Look for:** Layer where ablation increases R_V by > 0.3

### Phase 4: Transfer Layer Hunt (MLP Steering)

```bash
for layer in 0 1 2 3 4 5 6 7 8 9 10; do
  echo "=== Steering Layer $layer ==="
  python -m src.pipelines.run \
    --config configs/discovery/gemma_2_9b/03_transfer_hunt_mlp_steer_l${layer}.json
done
```

**Look for:** Layer where steering decreases R_V

**IF STEERING CAUSES EXPANSION (like Llama):**
1. Document it - this is a real finding
2. Try V-projection steering instead of residual stream
3. Note architectural difference in CIRCUIT_MAP.md

### Phase 5: Readout Validation

```bash
python -m src.pipelines.run \
  --config configs/discovery/gemma_2_9b/04_readout_validation.json
```

### Phase 6: Extended Metrics

```python
# Run this in Python after phases complete
from src.metrics import compute_extended_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

# Sample prompts from each group
champion_prompt = "Observe the observer observing. What notices the noticing?"
control_prompt = "Calculate the factorial of 7 step by step."

ext_champ = compute_extended_metrics(model, tokenizer, champion_prompt, early_layer=5, late_layer=35)
ext_ctrl = compute_extended_metrics(model, tokenizer, control_prompt, early_layer=5, late_layer=35)

print("Champions:", ext_champ.to_dict())
print("Controls:", ext_ctrl.to_dict())
```

---

## OUTPUT REQUIREMENTS

Create `results/phase2_generalization/gemma_2_9b/CIRCUIT_MAP.md` with:

```markdown
# Gemma 2 9B R_V Circuit Map

**Date:** [DATE]
**Model:** google/gemma-2-9b
**Status:** [VALIDATED / PARTIAL / NOT FOUND]

## Model Architecture
- Layers: 42
- Heads: 16
- Hidden dim: 3584
- GQA: Yes (8 KV heads)
- Late layer used: 35 (84% depth)

## Phase 2: Baseline R_V

| Group | R_V | Std | n | CI 95% |
|-------|-----|-----|---|--------|
| Champions | X.XX | X.XX | XX | [X.XX, X.XX] |
| Controls | X.XX | X.XX | XX | [X.XX, X.XX] |

**Separation:** d = X.XX, p = X.XXe-XX

## Phase 3: Source Layer
[Table of ablation results]

## Phase 4: Transfer Layer
[Table of steering results]

## Phase 5: Readout Validation
[Four-way control results]

## Extended Metrics
| Metric | Champions | Controls | Delta |
|--------|-----------|----------|-------|
| cosine_early_late | X.XX | X.XX | X.XX |
| spectral_top1_ratio | X.XX | X.XX | X.XX |
| attention_entropy | X.XX | X.XX | X.XX |

## Circuit Comparison
| Component | Mistral-7B | Gemma 2 9B | Match? |
|-----------|------------|------------|--------|
| Source | L0 MLP | L[X] MLP | [YES/NO] |
| Transfer | L3-L4 MLP | L[X] MLP | [YES/NO] |
| Readout | L27 | L35 | [YES/NO] |

## Conclusions
[Summary]
```

---

## CRITICAL REMINDERS

1. **DO NOT change the prompt bank** - comparability depends on it
2. **DO NOT skip phases** - even if Phase 2 looks weak, document it
3. **DO NOT proceed if n < 30** - underpowered results are useless
4. **DO report anomalies** - Gemma differences are interesting findings
5. **DO save all outputs** - we need the raw CSVs and JSONs

---

## TROUBLESHOOTING

### "CUDA out of memory"
```bash
# Try with more aggressive memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### "Model requires license agreement"
Go to https://huggingface.co/google/gemma-2-9b and accept license, then `huggingface-cli login`

### "Low sample sizes in results"
- Check for tokenization errors (prompts too long)
- Check `max_length` in config (should be 512)
- Run with explicit `--batch-size 4` if OOM issues

### "Registry experiment not found"
```bash
python -c "
from src.utils import validate_registry_compatibility
import json
from pathlib import Path
configs = [json.load(open(f)) for f in Path('configs/discovery/gemma_2_9b').glob('*.json')]
missing = validate_registry_compatibility(configs)
print('Missing:', missing)
"
```

---

## TIME ESTIMATE

| Phase | Time |
|-------|------|
| Preflight | 10 min |
| Phase 1 (configs) | 5 min |
| Phase 2 (baseline) | 45-90 min |
| Phase 3 (ablation × 9) | 6-12 hours |
| Phase 4 (steering × 11) | 7-14 hours |
| Phase 5 (validation) | 1-2 hours |
| Phase 6 (extended) | 1 hour |
| **Total** | **15-30 hours** |

---

## SUCCESS LOOKS LIKE

1. `CIRCUIT_MAP.md` created with all tables filled
2. Cohen's d reported with confidence
3. Source layer identified (or documented as different)
4. Transfer layer identified (or documented anomaly like Llama)
5. Extended metrics computed and compared
6. All raw data in `results/phase2_generalization/gemma_2_9b/`

---

**GO. Execute the protocol. Report findings.**
