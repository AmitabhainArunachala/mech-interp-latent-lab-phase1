# EXACT Mistral Methodology - Replicated for Pythia

## Source Code
**Original:** `models/mistral_7b_analysis.py`  
**Replicated:** `02_CODE/pythia_EXACT_MISTRAL_METHOD.py`

---

## EXACT Parameters (DO NOT CHANGE)

### Layer Selection
- **Early Layer:** 5 (fixed)
- **Late Layer:** 28 (for 32-layer models, ~84% depth)
- **Rationale:** Same as Mistral measurement

### Window Size
- **Window:** 16 tokens (last 16 tokens of prompt)
- **Rationale:** Same as Mistral measurement

### Value Extraction
- **Method:** Hook `v_proj` output during forward pass
- **Mistral:** `model.model.layers[layer_idx].self_attn.v_proj`
- **Pythia:** `model.gpt_neox.layers[layer_idx].attention.v_proj`
- **Rationale:** Extract value projections before attention computation

### Participation Ratio Computation

**Formula:** `PR = 1.0 / sum((s²/Σs²)²)`

**Process:**
1. Reshape value tensor: `[batch, seq_len, hidden]` → `[batch, seq_len, num_heads, d_v]`
2. Transpose: `[batch, heads, d_v, seq_len]`
3. For each head:
   - Extract last 16 tokens: `[d_v, 16]`
   - Compute SVD: `U, S, Vt = svd(v_window)`
   - Normalize squared singular values: `S_sq_norm = S² / sum(S²)`
   - Compute PR: `PR = 1.0 / sum(S_sq_norm²)`
4. Average PR across all heads

### R_V Computation

**Formula:** `R_V = PR_late / PR_early`

- **PR_early:** Participation ratio at layer 5
- **PR_late:** Participation ratio at layer 28
- **R_V < 1.0:** Contraction (dimensional collapse)
- **R_V > 1.0:** Expansion (dimensional increase)

---

## Tokenization Settings

```python
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512  # EXACT Mistral setting
).to(DEVICE)
```

---

## Model Loading Settings

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,  # EXACT Mistral setting
    output_hidden_states=True,
    attn_implementation="eager"  # EXACT Mistral setting
)
model.eval()
```

---

## Key Differences: Mistral vs Pythia

### Architecture Paths
- **Mistral:** `model.model.layers[layer_idx].self_attn.v_proj`
- **Pythia:** `model.gpt_neox.layers[layer_idx].attention.v_proj`

### Checkpoint Naming
- **Mistral:** Single checkpoint (final trained)
- **Pythia:** Multiple checkpoints:
  - `pythia-2.8b-deduped` (checkpoint 0)
  - `pythia-2.8b-step{num}` (intermediate)
  - `pythia-2.8b` (final, checkpoint 143k)

---

## Validation Checklist

Before running, verify:

- [ ] Early layer = 5
- [ ] Late layer = 28 (or 84% of total layers)
- [ ] Window size = 16 tokens
- [ ] Value extraction from `v_proj` output
- [ ] PR computation averages across heads
- [ ] R_V = PR_late / PR_early (not inverted)
- [ ] Tokenization: max_length=512, padding=True, truncation=True
- [ ] Model dtype: float16 (if CUDA available)
- [ ] attn_implementation: "eager"

---

## Expected Results (Mistral Baseline)

**Mistral-7B (from Phase 1):**
- L5 Recursive R_V: ~0.85
- Baseline R_V: ~1.00
- Contraction: 15.3%

**If Pythia matches:**
- L5 Recursive R_V: ~0.85
- Baseline R_V: ~1.00
- Contraction: ~15%

**If Pythia differs:**
- Document exact values
- Check measurement protocol
- Verify architecture differences

---

## Usage

```python
from pythia_EXACT_MISTRAL_METHOD import run_pythia_analysis, run_checkpoint_sweep

# Single checkpoint
results = run_pythia_analysis(prompts, output_path="results.csv")

# Checkpoint sweep
checkpoints = [0, 5000, 10000, 15000, 20000, 76000, 143000]
all_results = run_checkpoint_sweep(checkpoints, prompts)
```

---

*This methodology is EXACTLY replicated from Mistral measurement. Any changes invalidate comparison.*

