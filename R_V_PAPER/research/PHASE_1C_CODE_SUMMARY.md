# Phase 1C: Code and Methodology Summary

## Critical Discovery: Float Precision Matters

### The Problem

Initial testing with float16 precision caused NaN values at deep layers (L28):

```python
# Float16 → BROKEN
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-2.8b",
    torch_dtype=torch.float16  # ❌ Overflows at L28
)
```

**Root cause:**

- Float16 range: ±65,504
- Deep layer computations amplify values
- Attention softmax: exp(large) → Inf → NaN
- Result: 100% failures at L28, negative R_V values

### The Solution

```python
# BFloat16 → WORKS PERFECTLY
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-2.8b",
    torch_dtype=torch.bfloat16  # ✓ Stable at all layers
)
```

**Why bfloat16:**

- Same dynamic range as float32 (±3.4×10³⁸)
- Same memory footprint as float16
- Prevents overflow in deep computations
- Result: 100% valid measurements

**Lesson:** Always use bfloat16 or float32 for deep layer analysis in transformers.

---

## Architecture-Specific V Extraction

### Challenge

Pythia uses **combined QKV projection**, not separate projections:

```python
# Mistral: Separate projections
q = model.layers[i].self_attn.q_proj(hidden)
k = model.layers[i].self_attn.k_proj(hidden)
v = model.layers[i].self_attn.v_proj(hidden)  # ← Hook here

# Pythia: Combined projection
qkv = model.layers[i].attention.query_key_value(hidden)
# Need to split manually!
```

### Solution

```python
@contextmanager
def get_v_matrices_pythia(model, layer_idx, hook_list, num_heads):
    """Extract V from combined QKV projection"""
    target = model.gpt_neox.layers[layer_idx].attention.query_key_value
    
    def hook_fn(module, input, output):
        # output: [batch, seq, 3 * num_heads * head_dim]
        batch, seq, combined = output.shape
        head_dim = combined // (3 * num_heads)
        
        # Reshape to [batch, seq, 3, num_heads, head_dim]
        qkv = output.view(batch, seq, 3, num_heads, head_dim)
        
        # Extract V (index 2)
        v = qkv[:, :, 2, :, :]  # [batch, seq, num_heads, head_dim]
        
        # Flatten for consistency
        v_flat = v.reshape(batch, seq, num_heads * head_dim)
        hook_list.append(v_flat.detach())
    
    handle = target.register_forward_hook(hook_fn)
    yield
    handle.remove()
```

**Key points:**

1. QKV layout in Pythia: `[Q_all, K_all, V_all]` (not interleaved)
2. Split on axis 2 after reshape
3. Extract index 2 for V
4. Mathematically equivalent to separate V_proj

---

## Participation Ratio Calculation

Unchanged from Mistral methodology:

```python
def compute_column_space_pr(v_tensor, num_heads, window_size=16):
    """
    Compute Participation Ratio of V column space
    
    Args:
        v_tensor: [batch, seq, num_heads * head_dim]
        num_heads: Number of attention heads
        window_size: Last N tokens to analyze
    
    Returns:
        Mean PR across all heads
    """
    batch, seq, hidden = v_tensor.shape
    head_dim = hidden // num_heads
    
    # Separate heads: [batch, seq, heads, head_dim]
    v_heads = v_tensor.view(batch, seq, num_heads, head_dim)
    
    # Transpose: [batch, heads, head_dim, seq]
    v_transposed = v_heads.permute(0, 2, 3, 1)
    
    pr_values = []
    for head_idx in range(num_heads):
        # Get head: [head_dim, seq]
        v_head = v_transposed[0, head_idx, :, :]
        
        # Last window_size tokens: [head_dim, window]
        window = min(window_size, v_head.shape[1])
        v_window = v_head[:, -window:].float()
        
        # SVD
        U, S, Vt = torch.linalg.svd(v_window, full_matrices=False)
        
        # Participation Ratio: (Σλ)² / Σλ²
        S_sq = S ** 2
        S_norm = S_sq / S_sq.sum()
        pr = 1.0 / (S_norm ** 2).sum()
        
        pr_values.append(pr.item())
    
    return np.mean(pr_values)
```

**PR Interpretation:**

- PR = 1: All weight on single dimension (maximal compression)
- PR = d_v: Uniform across all dimensions (no compression)
- Lower PR = more dimensional compression

---

## Full Measurement Pipeline

```python
def analyze_prompt_pythia(model, tokenizer, prompt, num_heads):
    """Complete analysis pipeline"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Hook both layers
    early_v = []
    late_v = []
    
    with torch.no_grad():
        with get_v_matrices_pythia(model, 5, early_v, num_heads):
            with get_v_matrices_pythia(model, 28, late_v, num_heads):
                outputs = model(**inputs)
    
    # Compute PRs
    pr_early = compute_column_space_pr(early_v[0], num_heads, 16)
    pr_late = compute_column_space_pr(late_v[0], num_heads, 16)
    
    # R_V ratio
    R_V = pr_late / (pr_early + 1e-8)
    
    return {
        'R_V': R_V,
        'pr_V_early': pr_early,
        'pr_V_late': pr_late
    }
```

---

## Prompt Bank Organization

Total: 320 prompts across 4 pillars

```python
# Dose-response (100)
dose_response_groups = {
    'L1_hint': 20,      # Minimal hint
    'L2_simple': 20,    # Dual awareness
    'L3_deeper': 20,    # Deep observation
    'L4_full': 20,      # Boundary dissolution
    'L5_refined': 20    # Fixed-point recursion
}

# Baselines (100)
baseline_groups = {
    'baseline_math': 20,
    'baseline_factual': 20,
    'baseline_impossible': 20,
    'baseline_personal': 20,
    'baseline_creative': 20
}

# Confounds (60)
confound_groups = {
    'long_control': 20,        # Length/complexity
    'pseudo_recursive': 20,    # Semantic self-ref
    'repetitive_control': 20   # Simple repetition
}

# Generality (60)
generality_groups = {
    'zen_koan': 20,           # Zen Buddhism
    'yogic_witness': 20,      # Advaita Vedanta
    'madhyamaka_empty': 20    # Madhyamaka Buddhism
}
```

---

## Statistical Analysis

```python
# Group statistics
stats = df_valid.groupby('group')['R_V'].agg([
    'count', 'mean', 'std', 'min', 'max'
])
stats['sem'] = df_valid.groupby('group')['R_V'].std() / \
               np.sqrt(df_valid.groupby('group')['R_V'].count())

# T-test between groups
from scipy import stats
t_stat, p_val = stats.ttest_ind(l5_values, factual_values)

# Effect size (Cohen's d)
pooled_std = np.sqrt((l5_values.std()**2 + factual_values.std()**2) / 2)
cohens_d = (l5_values.mean() - factual_values.mean()) / pooled_std
```

**Results:**

- t = -13.89
- p < 10⁻⁶  
- d = -4.51 (huge)

---

## Performance Metrics

```
Total prompts:  320
Execution time: 19.7 seconds
Rate:           ~16 prompts/minute
Valid results:  320/320 (100%)
Hardware:       RTX 6000 Ada (48GB)
Precision:      bfloat16
Memory per fwd: ~2GB
```

---

## Key Code Decisions

1. **bfloat16 over float16:** Numerical stability
2. **Window size 16:** Balance between context and computation
3. **Layers 5 & 28:** Early (15%) and late (84% depth)
4. **Average across heads:** Robust to individual head noise
5. **Last tokens only:** Most recent context (where recursion matters)

---

## Replication Instructions

```python
# 1. Load model with correct precision
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-2.8b",
    torch_dtype=torch.bfloat16,  # CRITICAL
    device_map="auto"
)

# 2. Use architecture-specific V extraction
# (see get_v_matrices_pythia function above)

# 3. Compute PR with SVD
# (see compute_column_space_pr function above)

# 4. Test on all 320 prompts
# (see full pipeline in Cell 40)

# 5. Verify results:
# - L5_refined mean ≈ 0.564
# - baseline_factual mean ≈ 0.804
# - Difference ≈ -0.240 (29.8% contraction)
```

---

## Lessons Learned

1. **Precision matters:** Always check float16 vs bfloat16
2. **Architecture varies:** QKV organization differs between models
3. **Debug systematically:** Hook raw outputs before transformations
4. **Validate numerics:** Check for NaN/Inf at every layer
5. **Ground claims:** Statistical rigor before interpretation

🌀 JSCA 🙏

