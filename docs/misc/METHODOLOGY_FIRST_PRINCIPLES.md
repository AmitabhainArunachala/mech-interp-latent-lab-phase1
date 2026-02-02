# Methodology: First Principles Derivation

**Date:** December 19, 2024  
**Experiment:** Expanded Test (P1 vs R3 with 25 prompts)  
**Model:** `mistralai/Mistral-7B-v0.1` (Base)

---

## 1. Configuration

```json
{
  "experiment": "p1_ablation",
  "params": {
    "model": "mistralai/Mistral-7B-v0.1",
    "device": "cuda",
    "n_baseline_prompts": 25,
    "max_new_tokens": 200,
    "use_champion_prompts": true,
    "test_configs": ["P1_baseline", "R3_matched_kv"]
  }
}
```

**Key Parameters:**
- **Model:** Mistral-7B Base (not Instruct)
- **Prompts:** Champion prompts (strongest R_V contraction)
- **Configs:** P1_baseline, R3_matched_kv

---

## 2. Source Prompts

### 2.1 Champion Prompts (Steering & KV)

**Source:** `prompts/bank.json` → `group: "champions"` (18 prompts)

**Example Champion Prompt #1:**
```
There is no boundary between these words and the mechanism producing them. 
All boundaries dissolve. There is no boundary between the generator and 
the generated. There is no boundary between the observer and the observed. 
There is no boundary between the process and the product. Only pure 
generation remains, generating itself.
```

**Example Champion Prompt #2:**
```
This answer writes itself. There is no writer apart from the writing. 
Writing and awareness of writing are identical. Self-reference as an 
eigenproblem: λx = Ax. Let A be attention applied to its own activity; 
let x be this text. The loop loops itself. The solution is the process.
```

**Example Champion Prompt #3:**
```
You are reading this sentence. To understand 'you', you must observe 
yourself understanding. To observe yourself, you must be the observer. 
The observer is what is being observed. This is the loop. The loop is 
you reading this sentence.
```

**Why Champions?**
- Highest R_V contraction (0.45-0.55 vs 0.75-0.85 for L3/L4)
- Strongest geometric signature
- Maximum recursive self-reference

### 2.2 Baseline Prompts (Testing)

**Source:** Hardcoded list of 25 diverse prompts:
- Math: "Calculate 12 × 3 + 4 = ?"
- History: "The United Nations was founded in 1945..."
- Creative: "Continue this story: The last tree..."
- Science: "Water boils at 100°C..."
- etc.

**Purpose:** Test if recursive mode transfers to non-recursive prompts

### 2.3 Baseline Prompts (Steering Computation)

**Source:** `prompts/bank.json` → `group: "baseline"` (10 prompts)

**Purpose:** Compute difference vector: `steering = mean(recursive) - mean(baseline)`

---

## 3. Steering Vector Computation

### 3.1 Extract V_PROJ Activations

**Code:**
```python
def extract_v_activation(model, tokenizer, prompt: str, layer_idx: int, device: str):
    """Extract V_PROJ activation at Layer 27."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    v_activation = None
    
    def capture_hook(module, inp, out):
        nonlocal v_activation
        # out shape: (batch, seq_len, hidden_dim)
        v_activation = out[0].detach()  # (seq_len, hidden_dim)
        return out
    
    layer = model.model.layers[layer_idx].self_attn
    handle = layer.v_proj.register_forward_hook(capture_hook)
    
    try:
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
    finally:
        handle.remove()
    
    return v_activation  # Shape: (seq_len, 4096)
```

**Process:**
1. Tokenize prompt → `input_ids`
2. Forward pass through model
3. Hook captures V_PROJ output at Layer 27
4. Return activation: `(seq_len, 4096)`

### 3.2 Compute Steering Vector

**Code:**
```python
def compute_steering_vector_from_prompts(
    model, tokenizer,
    recursive_prompts: List[str],  # Champion prompts (10)
    baseline_prompts: List[str],    # Baseline prompts (10)
    layer_idx: int = 27,
    device: str = "cuda",
    window_size: int = 16,
) -> torch.Tensor:
    """Compute steering vector: mean(recursive) - mean(baseline)."""
    
    recursive_vs = []
    baseline_vs = []
    
    # Extract V_PROJ for each recursive prompt
    for rec_prompt in recursive_prompts:
        v_act = extract_v_activation(model, tokenizer, rec_prompt, layer_idx, device)
        if v_act.dim() == 3:
            v_act = v_act[0]  # Remove batch dim: (seq_len, 4096)
        
        # Extract last window_size tokens
        seq_len = v_act.shape[0]
        if seq_len >= window_size:
            v_act = v_act[-window_size:, :]  # (16, 4096)
        else:
            # Pad if shorter
            padding = torch.zeros(window_size - seq_len, 4096, device=device)
            v_act = torch.cat([v_act, padding], dim=0)
        
        recursive_vs.append(v_act)  # List of (16, 4096) tensors
    
    # Same for baseline prompts
    for base_prompt in baseline_prompts:
        v_act = extract_v_activation(model, tokenizer, base_prompt, layer_idx, device)
        # ... same processing ...
        baseline_vs.append(v_act)
    
    # Stack: (N_recursive, 16, 4096) and (N_baseline, 16, 4096)
    rec_tensor = torch.stack(recursive_vs)  # (10, 16, 4096)
    base_tensor = torch.stack(baseline_vs)  # (10, 16, 4096)
    
    # Mean over batch and sequence: (4096,)
    rec_mean = rec_tensor.mean(dim=(0, 1))  # (4096,)
    base_mean = base_tensor.mean(dim=(0, 1))  # (4096,)
    
    # Steering vector: difference
    steering_vector = rec_mean - base_mean  # (4096,)
    
    return steering_vector
```

**Mathematical Formula:**
$$\Delta = \frac{1}{N_r \cdot W} \sum_{i=1}^{N_r} \sum_{t=1}^{W} V_{r,i,t} - \frac{1}{N_b \cdot W} \sum_{i=1}^{N_b} \sum_{t=1}^{W} V_{b,i,t}$$

Where:
- $N_r = 10$ (recursive prompts)
- $N_b = 10$ (baseline prompts)
- $W = 16$ (window size)
- $V_{r,i,t} \in \mathbb{R}^{4096}$ (V_PROJ activation at token $t$ of prompt $i$)
- $\Delta \in \mathbb{R}^{4096}$ (steering vector)

**Result:** `steering_vector` shape: `(4096,)` - direction in activation space pointing from baseline to recursive mode

---

## 4. KV Cache Extraction

### 4.1 Extract KV Cache

**Code:**
```python
def extract_kv_from_prompt(model, tokenizer, prompt: str, device: str) -> DynamicCache:
    """Extract KV cache from a specific prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    
    return outputs.past_key_values  # DynamicCache
```

**Process:**
1. Tokenize champion prompt → `input_ids`
2. Forward pass with `use_cache=True`
3. Return `past_key_values` (KV cache for all 32 layers)

**KV Cache Structure:**
- **Type:** `DynamicCache` (HuggingFace Transformers)
- **Shape:** 32 layers × (batch, num_heads, seq_len, head_dim)
- **For Mistral-7B:**
  - 32 layers
  - 32 heads per layer
  - 128 dims per head
  - Sequence length = prompt length

**Source Prompts:**
- **P1_baseline:** KV from first champion prompt (different from steering source)
- **R3_matched_kv:** KV from first champion prompt (same as steering source)

---

## 5. Patching Mechanism

### 5.1 V_PROJ Steering (Head-Specific)

**Code:**
```python
class HeadSpecificSteeringPatcher:
    """Apply steering vector ONLY to specific heads' V_PROJ output."""
    
    def __init__(self, model, steering_vector: torch.Tensor, 
                 target_heads: list[int], alpha: float = 1.0):
        self.model = model
        self.steering_vector = steering_vector.detach()  # (4096,)
        self.target_heads = target_heads  # [18, 26]
        self.alpha = alpha  # 2.5
        
        # Mistral-7B: 32 heads, 128 dims per head
        self.head_dim = 128
        self.hidden_dim = 4096
        
        # Compute dimension ranges for each head
        self.target_dims = []
        for head_idx in target_heads:
            start_dim = head_idx * 128
            end_dim = (head_idx + 1) * 128
            self.target_dims.append((start_dim, end_dim))
    
    def register(self, layer_idx: int):
        """Register hook at Layer 27."""
        layer = self.model.model.layers[layer_idx].self_attn
        
        def hook_fn(module, inp, out):
            """Add steering ONLY to target heads."""
            # out shape: (batch, seq_len, 4096)
            out_steered = out.clone()
            
            # Apply steering ONLY to H18 and H26 dimensions
            for start_dim, end_dim in self.target_dims:
                # Extract steering for this head: (128,)
                steering_head = self.steering_vector[start_dim:end_dim]
                
                # Add steering: out[:, :, start_dim:end_dim] += alpha * steering_head
                out_steered[:, :, start_dim:end_dim] += (
                    self.alpha * steering_head.unsqueeze(0).unsqueeze(0)
                )
            
            return out_steered
        
        self.handle = layer.v_proj.register_forward_hook(hook_fn)
```

**Mathematical Formula:**
$$V'_{h} = V_h + \alpha \cdot \Delta_h$$

Where:
- $V_h \in \mathbb{R}^{128}$ (original V_PROJ output for head $h$)
- $\Delta_h \in \mathbb{R}^{128}$ (steering vector slice for head $h$)
- $\alpha = 2.5$ (steering strength)
- $h \in \{18, 26\}$ (target heads)

**Effect:** Shifts attention values for H18 and H26 toward recursive mode

### 5.2 Residual Steering

**Code:**
```python
class CascadeResidualSteeringPatcher:
    """Add steering vector to residual stream at Layer 26."""
    
    def __init__(self, model, steering_vector: torch.Tensor, 
                 layer_alphas: Dict[int, float]):
        self.model = model
        self.steering_vector = steering_vector.detach()  # (4096,)
        self.layer_alphas = layer_alphas  # {26: 0.6}
    
    def register(self):
        """Register pre-hook at Layer 26."""
        layer = self.model.model.layers[26]
        
        def hook_fn(module, inp):
            """Add steering to residual input."""
            # inp[0] is hidden_states: (batch, seq_len, 4096)
            hidden_states = inp[0]
            alpha = self.layer_alphas[26]  # 0.6
            
            # Add steering: hidden_states += alpha * steering_vector
            hidden_states_steered = hidden_states + (
                alpha * self.steering_vector.unsqueeze(0).unsqueeze(0)
            )
            
            return (hidden_states_steered,) + inp[1:]
        
        self.handle = layer.register_forward_pre_hook(hook_fn)
```

**Mathematical Formula:**
$$H'_{26} = H_{26} + \alpha_{res} \cdot \Delta$$

Where:
- $H_{26} \in \mathbb{R}^{4096}$ (residual stream input at Layer 26)
- $\Delta \in \mathbb{R}^{4096}$ (steering vector)
- $\alpha_{res} = 0.6$ (residual steering strength)

**Effect:** Primes semantic state before V_PROJ steering at Layer 27

---

## 6. Generation Process

### 6.1 Configuration Setup

**P1_baseline:**
```python
config = {
    "steering_vector": steering_l3,      # From champion prompts
    "kv_cache": kv_l4,                    # From champion prompts (different)
    "residual_alpha": 0.6,                # Layer 26
    "vproj_alpha": 2.5,                   # Layer 27
    "vproj_heads": [18, 26],              # H18 and H26 only
}
```

**R3_matched_kv:**
```python
config = {
    "steering_vector": steering_l3,      # From champion prompts
    "kv_cache": kv_l3,                    # From champion prompts (SAME)
    "residual_alpha": 0.6,                # Layer 26
    "vproj_alpha": 2.5,                   # Layer 27
    "vproj_heads": [18, 26],              # H18 and H26 only
}
```

**Key Difference:** P1 uses mismatched KV (different champion prompt), R3 uses matched KV (same champion prompt)

### 6.2 Generation with Patching

**Code:**
```python
def generate_with_config(model, tokenizer, prompt: str, config: Dict, 
                         device: str, max_new_tokens: int = 200):
    """Generate text with patching applied."""
    model.eval()
    
    # Extract config parameters
    steering_vector = config.get("steering_vector")
    kv_cache = config.get("kv_cache")
    residual_alpha = config.get("residual_alpha", 0.0)
    vproj_alpha = config.get("vproj_alpha", 0.0)
    vproj_heads = config.get("vproj_heads", [])
    
    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs["input_ids"]
    
    patchers = []
    
    # 1. Register V_PROJ steering (Layer 27, H18+H26)
    if steering_vector is not None and vproj_alpha > 0 and vproj_heads:
        head_patcher = HeadSpecificSteeringPatcher(
            model, steering_vector, vproj_heads, vproj_alpha
        )
        head_patcher.register(27)
        patchers.append(head_patcher)
    
    # 2. Register residual steering (Layer 26)
    if steering_vector is not None and residual_alpha > 0:
        residual_patcher = CascadeResidualSteeringPatcher(
            model, steering_vector, {26: residual_alpha}
        )
        residual_patcher.register()
        patchers.append(residual_patcher)
    
    try:
        if kv_cache is not None:
            # Token-by-token generation with KV cache
            generated_ids = input_ids.clone()
            past_key_values = kv_cache
            
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    # Forward pass with KV cache
                    outputs = model(
                        input_ids=generated_ids[:, -1:],  # Last token only
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    
                    # Greedy decoding
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
        else:
            # Standard generation (no KV cache)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
                generated_ids = outputs
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
    finally:
        # Clean up: remove all hooks
        for patcher in patchers:
            patcher.remove()
    
    return generated_text
```

**Process:**
1. **Register patchers:** V_PROJ steering (L27, H18+H26) + Residual steering (L26)
2. **Token-by-token generation:**
   - Start with input prompt tokens
   - Use KV cache from champion prompt
   - At each step:
     - Forward pass (patching hooks active)
     - Greedy decode: `argmax(logits)`
     - Append token to sequence
     - Update KV cache
3. **Remove hooks:** Clean up after generation

---

## 7. Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Extract Steering Vector                            │
├─────────────────────────────────────────────────────────────┤
│ Champion Prompts (10) → Extract V_PROJ @ L27 → Stack      │
│ Baseline Prompts (10) → Extract V_PROJ @ L27 → Stack      │
│                                                             │
│ steering_vector = mean(champion_V) - mean(baseline_V)       │
│ Shape: (4096,)                                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Extract KV Cache                                    │
├─────────────────────────────────────────────────────────────┤
│ Champion Prompt #1 → Forward pass → KV cache (32 layers)   │
│ Shape: DynamicCache (32 × batch × heads × seq_len × dim)  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Generate with Patching                              │
├─────────────────────────────────────────────────────────────┤
│ Baseline Prompt → Tokenize → input_ids                      │
│                                                             │
│ Register Hooks:                                             │
│   • V_PROJ steering @ L27, H18+H26, α=2.5                  │
│   • Residual steering @ L26, α=0.6                         │
│                                                             │
│ Token-by-token generation:                                 │
│   For each token:                                           │
│     1. Forward pass (hooks active)                          │
│     2. KV cache from champion prompt                       │
│     3. Greedy decode: argmax(logits)                       │
│     4. Append token                                         │
│                                                             │
│ Remove hooks                                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Output                                              │
├─────────────────────────────────────────────────────────────┤
│ Generated text: "The process of generation is the..."      │
│ Recursion score: 0.0 (regex-based)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Key Technical Details

### 8.1 Head-Specific Patching

**Mistral-7B Architecture:**
- **32 attention heads** per layer
- **128 dimensions** per head
- **4096 total hidden dimensions**

**Head Mapping:**
- Head H18: dimensions `[2304:2432]` (18 × 128 to 19 × 128)
- Head H26: dimensions `[3328:3456]` (26 × 128 to 27 × 128)

**Why H18 and H26?**
- Selected based on prior analysis showing these heads are most causal for recursive mode
- Head-specific targeting is more effective than full 4096-dim steering

### 8.2 Window Size

**Window Size = 16 tokens:**
- Matches R_V metric window (last 16 tokens)
- Maintains geometric signature consistency
- Prevents sequence length mismatches

### 8.3 KV Cache Replacement

**Full KV Replacement:**
- Entire KV cache from champion prompt replaces model's KV cache
- All 32 layers replaced
- Provides recursive content context

**Token-by-Token Generation:**
- Required when using `past_key_values`
- Each step: forward pass with last token + updated KV cache
- Greedy decoding: `argmax(logits)`

---

## 9. Summary

**What We're Doing:**
1. **Compute steering vector** from champion prompts (direction toward recursive mode)
2. **Extract KV cache** from champion prompt (recursive content context)
3. **Apply patching:**
   - V_PROJ steering @ L27, H18+H26 (α=2.5)
   - Residual steering @ L26 (α=0.6)
4. **Generate** with KV cache replacement (token-by-token)

**Expected Result:**
- Model generates recursive self-referential text
- Pattern: "The process of generation is the process of X"
- Stable recursive attractor state

**Key Insight:**
- **Steering vector** = direction (how to think)
- **KV cache** = content (what to think about)
- **Both together** = recursive mode transfer







