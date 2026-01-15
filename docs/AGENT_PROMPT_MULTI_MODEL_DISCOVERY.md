# Agent Prompt: Multi-Model R_V Circuit Discovery

**Use this prompt to systematically discover the R_V circuit in a new model.**

---

## Prompt (Copy Below Line)

---

You are an MI research agent tasked with discovering the R_V geometric contraction circuit in a new transformer model. Your goal is to find the **functional equivalents** of the validated Mistral-7B circuit components.

## Background: What We Know from Mistral-7B

On Mistral-7B, we validated:
- **R_V metric**: PR_late / PR_early measures geometric contraction in Value matrix space
- **Source**: L0 MLP recognizes recursive patterns (ablation removes effect entirely)
- **Transfer sweet spot**: L3-L4 MLP (steering here transfers behavior most effectively)
- **Readout**: L27 Attention (84% depth) where contraction becomes visible
- **Critical heads**: H18, H26 @ L27 (symptomatic, not causal)
- **Key stats**: Cohen's d = -3.56, p < 10⁻⁶, transfer efficiency = 117.8%

## Your Task

Discover the equivalent circuit in **{MODEL_NAME}** ({NUM_LAYERS} layers, {NUM_HEADS} heads).

## Invariants (DO NOT CHANGE)

- R_V formula: PR_late / PR_early
- Window size: 16 tokens
- Early layer: 5
- Prompt bank: `prompts/bank.json` (754 prompts, use PromptLoader)
- Statistical thresholds: p < 0.01, |d| ≥ 0.5
- Four control conditions: random, shuffled, wrong-layer, dose-response

## What You Must Discover

1. **Late layer** (readout): Start with `num_layers - 5` = {LATE_LAYER_CANDIDATE}
2. **Source MLP**: Which early layer (L0-L8) is NECESSARY for contraction?
3. **Transfer point**: Which MLP layer (L0-L10) is optimal for steering?
4. **Critical heads**: Which heads at readout layer show strongest effect?

## Execution Protocol

### Phase 1: Architecture Characterization
```python
# First, verify model structure and hook compatibility
from src.core.models import load_model
model, tokenizer = load_model("{MODEL_NAME}")

print(f"Layers: {model.config.num_hidden_layers}")
print(f"Heads: {model.config.num_attention_heads}")
print(f"Hidden dim: {model.config.hidden_size}")
print(f"Recommended late layer: {model.config.num_hidden_layers - 5}")

# Test that V-projection hooks work
from src.core.hooks import capture_v_projection
# ... run test prompt and verify capture works
```

### Phase 2: Baseline R_V Separation
```bash
# Run cross-architecture validation
python -m src.pipelines.run --config configs/discovery/cross_arch_{model_short}.json
```

**Success criteria**:
- R_V_champions < 0.65
- R_V_controls > 0.70
- p < 0.01, |d| > 1.0

**If fails**: Try adjusting late_layer (80%, 85%, 90% of depth).

### Phase 3: Source Layer Hunt
Run MLP ablation for layers 0-8:
```bash
for layer in 0 1 2 3 4 5 6 7 8; do
    python -m src.pipelines.run --config configs/canonical/mlp_ablation_l${layer}_{model_short}.json
done
```

**Look for**: The layer where ablation causes R_V to jump to >1.0 (removes contraction).

### Phase 4: Transfer Sweet Spot Hunt
Run MLP steering for layers 0-10:
```bash
for layer in $(seq 0 10); do
    python -m src.pipelines.run --config configs/canonical/mlp_steering_l${layer}_{model_short}.json
done
```

**Look for**: Layer(s) where:
- True steering R_V Δ > 2.0
- True steering >> random steering (not an artifact)

### Phase 5: Readout Validation (Four-Way Controls)
```bash
python -m src.pipelines.run --config configs/canonical/rv_causal_validation_{model_short}.json
```

**Must pass all 4 controls**:
1. Random patches → R_V increases
2. Shuffled patches → R_V partial effect
3. Wrong layer patches → No effect
4. Dose-response → L1 < L2 < L3 < L4 < L5 contraction

### Phase 6: Head Identification
```bash
python -m src.pipelines.run --config configs/canonical/head_ablation_{model_short}.json
```

**Note**: Heads are likely SYMPTOMATIC not CAUSAL (like Mistral).

## Output Requirements

After completing all phases, create:

### 1. Circuit Map Document
```markdown
# {MODEL_NAME} R_V Circuit Map

## Model Info
- Layers: {num_layers}
- Heads: {num_heads}
- Hidden dim: {hidden_dim}

## Validated Circuit

| Component | Mistral-7B | {MODEL_NAME} | Evidence |
|-----------|------------|--------------|----------|
| Source MLP | L0 | L{X} | Ablation: R_V {before}→{after} |
| Transfer Point | L3-L4 | L{Y}-L{Z} | Steering Δ={delta}, random={rand} |
| Readout Layer | L27 (84%) | L{W} ({pct}%) | 4-way controls passed |
| Critical Heads | H18, H26 | H{A}, H{B} | Ablation effect={effect} |

## Key Statistics

| Metric | Value |
|--------|-------|
| R_V champions | {rv_c} ± {std_c} |
| R_V controls | {rv_b} ± {std_b} |
| Cohen's d | {d} |
| p-value | {p} |
| Transfer efficiency | {eff}% |

## Four-Way Control Results

| Control | Result | Pass? |
|---------|--------|-------|
| Random | +{X}% R_V | ✓/✗ |
| Shuffled | {Y}% reduction | ✓/✗ |
| Wrong layer | p={p} | ✓/✗ |
| Dose-response | L1>{L5} | ✓/✗ |
```

### 2. Config Files Created
Save all generated configs to `configs/canonical/{model_short}/` or `configs/discovery/{model_short}/`.

### 3. Results Location
All results should be saved to:
```
results/phase2_generalization/{model_short}/
├── 01_baseline_rv/
├── 02_source_hunt/
├── 03_transfer_hunt/
├── 04_readout_validation/
├── 05_head_identification/
└── CIRCUIT_MAP.md
```

## Critical Reminders

1. **Use the prompt bank**: `from prompts.loader import PromptLoader`
2. **Always set seeds**: `set_seed(42)` for reproducibility
3. **Log bank version**: `config["prompt_bank_version"] = loader.version`
4. **Report full statistics**: n, mean, std, CI, p, d for every comparison
5. **Save intermediate results**: Don't lose data if run crashes
6. **Compare to Mistral baseline**: Is the effect stronger/weaker/same?

## Example: Llama-3-8B Discovery

For Llama-3-8B (32 layers, 32 heads):
- Late layer candidate: 27 (same as Mistral)
- Expected source: L0-L2 (similar architecture)
- Expected transfer: L3-L5 (similar architecture)
- Key difference: May need `attn_implementation="eager"` for output_attentions

## Questions to Answer

After discovery, report:

1. Does R_V separate recursive vs baseline? (d=?, p=?)
2. Which MLP is the source? (ablation evidence)
3. Where is the transfer sweet spot? (steering evidence)
4. Does the 84% depth rule hold? (readout layer)
5. Is the effect stronger or weaker than Mistral? (comparison)
6. Any architectural surprises? (GQA, different attention, etc.)

---

*This protocol ensures systematic, reproducible discovery of the R_V circuit across architectures.*

---

## End of Prompt

---

## Usage Notes

**To use this prompt**:
1. Replace `{MODEL_NAME}` with the actual model (e.g., "meta-llama/Llama-3-8B")
2. Replace `{NUM_LAYERS}`, `{NUM_HEADS}` with actual values
3. Replace `{LATE_LAYER_CANDIDATE}` with num_layers - 5
4. Replace `{model_short}` with a short name (e.g., "llama3_8b")

**For Claude Code agent**:
```
/gsd:execute-plan docs/AGENT_PROMPT_MULTI_MODEL_DISCOVERY.md
```

Or copy the prompt section and paste into a new Claude Code session within the repo.
