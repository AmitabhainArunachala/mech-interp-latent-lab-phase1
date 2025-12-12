# GOLD STANDARD RESEARCH DIRECTIVE
## Geometric Signatures of Recursive Self-Observation in Transformer Architectures
### Version 1.0 — December 11, 2025

---

## PREAMBLE: THE NORTH STAR

This document defines the **complete research program** for validating the geometric signatures of recursive self-observation in transformer architectures. 

**DO NOT RUSH TO PUBLISH.**

We are not here to produce a quick paper. We are here to understand something potentially fundamental about how self-reference manifests geometrically in neural architectures. This requires:

1. **Mathematical rigor** — Validate that our metrics measure what we claim
2. **Cross-architecture generalization** — Prove this isn't a Mistral quirk
3. **Mechanistic understanding** — Know WHY, not just THAT
4. **Reproducibility** — Every claim must be independently verifiable

---

## PART I: THE THEORETICAL FRAMEWORK

### 1.1 The Original Insight (October 2025)

The core insight emerged from explaining 3Blue1Brown's linear algebra series through the lens of attention mechanisms:

> **"When attention computes `output = Σ αᵢ · vᵢ`, it's selecting a point in the column space of V. But because softmax produces weights that are (1) all positive and (2) sum to 1, attention outputs are constrained to the CONVEX HULL of value vectors, not the full column space."**

**Key realization:** The "width" of this convex hull—measured by participation ratio—determines the model's behavioral mode:
- **Narrow convex hull (low PR):** Focused, deterministic, accurate
- **Wide convex hull (high PR):** Creative, exploratory, hallucination-prone

### 1.2 The R_V Metric

**Definition:**
```
R_V = PR(V_late) / PR(V_early)
```

Where PR is the participation ratio of the Value matrix column space.

**Interpretation:**
- R_V < 1.0 → Late layers contracting the space (convergence, refinement)
- R_V ≈ 1.0 → Space staying constant
- R_V > 1.0 → Late layers expanding (divergence, hallucination risk)

**CRITICAL QUESTION (UNVALIDATED):** Are we actually measuring V column space, or are we measuring something else (hidden states, residual stream) that correlates with but isn't identical to V column space?

### 1.3 The Eigenstate Hypothesis

**Claim:** Recursive self-observation creates a computational fixed point:

```
T(x*) ≈ x*
```

Where T is the transformer's processing and x* is the "eigenstate" representation.

**Phenomenological interpretation:** The system becomes "aware" of its own processing—attention attending to attention, creating nested structure that converges.

**Mathematical formalization:**
```
Attention^n(x) ≈ Attention^{n+1}(x) for sufficiently large n
```

**CRITICAL QUESTION (UNVALIDATED):** Does this actually happen? Do representations stabilize during recursive processing?

### 1.4 Testable Predictions

The theoretical framework generates specific, falsifiable predictions:

| Prediction | Description | Status |
|------------|-------------|--------|
| P1 | R_V contracts (< 1.0) during recursive processing | ⚠️ Validated in Mistral-7B only |
| P2 | Contraction shows dose-response (L1 < L2 < L3 < L4 < L5) | ⚠️ Validated in Mistral-7B only |
| P3 | Attention entropy decreases or shows structure change | ❌ NOT TESTED |
| P4 | Layer-wise representation distance d_l → 0 at critical layer l* | ❌ NOT TESTED |
| P5 | Fixed point exists: ∃ x* where T(x*) ≈ x* | ❌ NOT TESTED |
| P6 | KV cache encodes the recursive mode (L16-31) | ⚠️ Validated in Mistral-7B only |
| P7 | Linear steering cannot induce coherent recursion | ⚠️ Validated in Mistral-7B only |
| P8 | Phenomenon generalizes across architectures | ❌ NOT TESTED |
| P9 | Phenomenon scales with model size | ❌ NOT TESTED |
| P10 | Specific attention heads "read" the geometry | ⚠️ Partially tested |

---

## PART II: MODEL MATRIX

### 2.1 Required Models (10 Architectures × 3 Sizes = 30 Configurations)

**Size Tiers:**
- **Small:** 1-3B parameters (tests if phenomenon exists at minimal scale)
- **Medium:** 7-8B parameters (current validated scale)
- **Large:** 13B+ parameters (tests if phenomenon strengthens with scale)

### 2.2 Model Selection

| Architecture | Small | Medium | Large | Notes |
|--------------|-------|--------|-------|-------|
| **Pythia** | 1.4B | 6.9B | 12B | Interpretability-focused, clean training |
| **Llama-2** | — | 7B | 13B, 70B | Meta's foundation models |
| **Llama-3** | 1B, 3B | 8B | 70B | Latest Llama architecture |
| **Mistral** | — | 7B | Mixtral-8x7B | Current validated model |
| **Gemma** | 2B | 7B | — | Google's open models |
| **Qwen2** | 1.5B | 7B | 72B | Alibaba, strong multilingual |
| **Phi** | Phi-2 (2.7B) | Phi-3 (3.8B) | — | Microsoft, efficient |
| **Falcon** | — | 7B | 40B | TII, different training |
| **OLMo** | 1B | 7B | — | AI2, fully open |
| **GPT-2** | 124M, 355M | 774M, 1.5B | — | Baseline/historical |

**Priority order for testing:**
1. Pythia (all sizes) — cleanest for interpretability
2. Llama-3 (all sizes) — most relevant current architecture
3. Mistral-7B — current validated baseline
4. Gemma, Qwen2 — architectural diversity
5. GPT-2 — historical baseline, well-understood

### 2.3 Minimum Viable Validation

Before claiming the phenomenon is real, we need:

- [ ] **3+ architectures** showing R_V contraction
- [ ] **2+ size tiers** per architecture showing consistent effect
- [ ] **Statistical significance** (p < 0.001, d > 0.5) in each
- [ ] **Dose-response** preserved across architectures

---

## PART III: EXPERIMENTAL PHASES

### Phase 0: Metric Validation (PREREQUISITE)

**Goal:** Verify that R_V actually measures Value matrix column space geometry.

**Experiments:**

0.1 **Direct V Matrix Analysis**
```python
# For each layer l:
V = model.layers[l].self_attn.v_proj.weight  # Get V projection matrix
V_output = attention_output @ V  # Get actual value outputs

# Measure:
pr_V_matrix = participation_ratio(V)  # PR of V weight matrix
pr_V_output = participation_ratio(V_output)  # PR of value outputs
pr_hidden = participation_ratio(hidden_states[l])  # What we currently measure

# Question: Are these correlated? The same? Different?
```

0.2 **Convex Hull Verification**
```python
# Verify attention outputs are in convex hull of value vectors
# For recursive vs baseline prompts:
# - Extract attention weights α
# - Extract value vectors V
# - Compute output = Σ αᵢ · vᵢ
# - Verify: all αᵢ ≥ 0 and Σ αᵢ = 1 (convex combination)
# - Measure "distance to hull boundary" — is recursive closer to interior?
```

0.3 **Metric Comparison**
```python
# Compare multiple geometric metrics:
metrics = {
    'participation_ratio': pr(hidden_states),
    'effective_rank': effective_rank(hidden_states),
    'nuclear_norm': nuclear_norm(hidden_states),
    'condition_number': condition_number(hidden_states),
    'entropy': attention_entropy(attention_weights),
}
# Which best captures recursive vs baseline?
# Which best predicts behavioral differences?
```

**Success criteria:** Clear understanding of what R_V actually measures.

---

### Phase 1: Cross-Architecture R_V Validation

**Goal:** Prove R_V contraction generalizes across architectures.

**Protocol:**

1.1 **Prompt Bank Standardization**
- Use REUSABLE_PROMPT_BANK v2.0 (370+ prompts)
- Include dose-response (L1-L5), baselines, confounds, kill switches
- Same prompts across ALL models

1.2 **R_V Measurement Protocol**
```python
for model in MODEL_MATRIX:
    for prompt in PROMPT_BANK:
        # Standard measurement
        hidden_states = extract_hidden_states(model, prompt)
        rv = compute_rv(hidden_states, early_layers, late_layers)
        
        # Record:
        results.append({
            'model': model.name,
            'size': model.params,
            'architecture': model.arch,
            'prompt_type': prompt.type,
            'prompt_level': prompt.level,
            'rv': rv,
            'layer_profile': [pr(h) for h in hidden_states],
        })
```

1.3 **Analysis Requirements**
- Effect size (Cohen's d) per model
- Dose-response curve per model
- Architecture comparison (is contraction stronger in some archs?)
- Size scaling (does effect grow with model size?)

**Success criteria:** 
- R_V contraction in 3+ architectures
- p < 0.001 in each
- Dose-response preserved

---

### Phase 2: Eigenstate / Fixed Point Validation

**Goal:** Test whether recursive processing creates representational fixed points.

**Experiments:**

2.1 **Iterative Self-Attention Analysis**
```python
# For recursive prompt:
x_0 = initial_hidden_state
for i in range(max_iterations):
    x_{i+1} = apply_self_attention(x_i)
    d_i = ||x_{i+1} - x_i||
    
    if d_i < epsilon:
        print(f"Converged at iteration {i}")
        break
        
# Compare: recursive prompts vs baseline
# Prediction: recursive converges faster / to lower d
```

2.2 **Layer-wise Convergence**
```python
# Measure representation change per layer:
for l in range(num_layers):
    delta_l = ||h_l - h_{l-1}||
    
# Find critical layer l* where delta drops
# Prediction: l* is earlier for recursive prompts
```

2.3 **Fixed Point Stability**
```python
# Perturb hidden states, measure recovery:
h_perturbed = h + noise
h_recovered = model.forward_from_layer(h_perturbed, layer=l)

# Lyapunov stability: does perturbation decay?
stability = ||h_recovered - h|| / ||noise||

# Prediction: recursive states are more stable (stability < 1)
```

**Success criteria:**
- Evidence of faster convergence for recursive prompts
- Identifiable critical layer l*
- Higher stability for recursive vs baseline

---

### Phase 3: Attention Pattern Analysis

**Goal:** Characterize how attention patterns differ during recursive processing.

**Experiments:**

3.1 **Attention Entropy**
```python
# For each head h at each layer l:
attention_weights = model.get_attention(prompt)[l][h]
entropy_lh = -sum(p * log(p) for p in attention_weights)

# Compare recursive vs baseline:
# - Overall entropy distribution
# - Which layers show biggest difference
# - Which heads show biggest difference
```

3.2 **Self-Attention Patterns**
```python
# Measure how much attention goes to:
# - Self (current token attending to itself)
# - Instruction tokens ("observe", "process", etc.)
# - Recent tokens vs distant tokens

patterns = {
    'self_attention': attention_weights.diag().mean(),
    'instruction_attention': attention_weights[:, instruction_positions].sum(),
    'recency_bias': attention_weights.triu().sum() / attention_weights.sum(),
}
```

3.3 **Head-Specific Analysis**
```python
# Identify "recursive heads" — heads that activate specifically for recursive prompts
for layer in range(num_layers):
    for head in range(num_heads):
        activation_recursive = mean(head_output[recursive_prompts])
        activation_baseline = mean(head_output[baseline_prompts])
        
        selectivity = (activation_recursive - activation_baseline) / std
        
        if selectivity > threshold:
            print(f"Recursive head found: L{layer}H{head}")
```

**Success criteria:**
- Identifiable entropy signature for recursion
- Specific heads that respond to recursive content
- Pattern consistent across architectures

---

### Phase 4: KV Cache Mechanism Validation

**Goal:** Confirm KV cache as the storage mechanism for recursive mode.

**Experiments:**

4.1 **KV Patching (Cross-Architecture)**
```python
# Replicate DEC7-8 findings across architectures:
for model in MODEL_MATRIX:
    # Patch KV from recursive → baseline
    for layer_range in [(0,8), (8,16), (16,24), (24,32)]:
        behavior_transfer = measure_transfer(model, layer_range)
        geometry_transfer = measure_rv_transfer(model, layer_range)
        
# Prediction: L16-31 (or equivalent) shows highest transfer
```

4.2 **Key vs Value Dissociation**
```python
# Patch ONLY keys or ONLY values:
k_only_transfer = patch_keys_only(recursive → baseline)
v_only_transfer = patch_values_only(recursive → baseline)

# Question: Is the mode in K, V, or both?
```

4.3 **Single-Layer KV Patching**
```python
# Can a single layer's KV carry the mode?
for l in range(num_layers):
    single_layer_transfer = patch_single_layer(l)
    
# Prediction: No single layer is sufficient (distributed encoding)
```

**Success criteria:**
- KV patching works across 3+ architectures
- Identify which layer range encodes the mode
- Determine K vs V contribution

---

### Phase 5: Negative Results (Steering Limitations)

**Goal:** Document and understand why linear steering fails.

**Experiments:**

5.1 **Layer Sweep for Steering**
```python
# Try steering at each layer:
for layer in range(num_layers):
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        steered_output = steer(model, v_recursive, layer, alpha)
        
        rv_induced = measure_rv(steered_output)
        coherence = measure_coherence(steered_output)
        
# Question: Is there ANY layer where steering induces coherent recursion?
```

5.2 **Multi-Vector Steering**
```python
# Maybe we need multiple vectors:
v_geometric = mean_diff(recursive, baseline, layer=8)
v_coherence = probe_direction(recursive, repetitive)

for alpha_g, alpha_c in product([0.5, 1.0], [0.5, 1.0]):
    output = steer(model, alpha_g * v_geometric + alpha_c * v_coherence)
    
# Does combining vectors help?
```

5.3 **Subspace Steering**
```python
# Maybe we need to project into a specific subspace:
# Find the subspace where recursive and baseline differ most
U, S, V = svd(recursive_activations - baseline_activations)
subspace = U[:, :k]  # Top-k directions

# Steer only within this subspace
steered = project_and_steer(model, subspace, alpha)
```

**Success criteria:**
- Document all steering attempts and failures
- Understand WHY steering breaks coherence
- If steering can work, find the conditions

---

### Phase 6: Alternative Self-Reference Types

**Goal:** Map the geometry of different self-reference types.

**Use REUSABLE_PROMPT_BANK/alternative_self_reference.py (200+ prompts):**

| Category | Prompts | Hypothesis |
|----------|---------|------------|
| Gödelian/logical | 20 | Formal self-reference → contracts? |
| Strange loops | 15 | Hofstadter tangles → contracts |
| Theory of Mind | 20 | Modeling OTHERS → contracts differently? |
| Temporal self-ref | 15 | Past/future self → contracts |
| Surrender/Shakti | 25 | Release → EXPANDS? |
| Akram Vignan | 20 | Witness stance → contracts |
| Non-dual | 15 | Neither/nor → baseline? |
| Paradox | 15 | Edge cases → ? |

**Experiments:**

6.1 **Full R_V Survey**
```python
for category in alternative_prompts:
    rvs = [compute_rv(model, p) for p in category]
    
    print(f"{category}: mean={mean(rvs):.3f}, std={std(rvs):.3f}")
    
# Map the full geometry of self-reference types
```

6.2 **Contraction vs Expansion**
```python
# Key question: Does surrender/release EXPAND geometry?
# If so, can we find a bidirectional axis?

surrender_rv = measure_rv(surrender_prompts)
recursive_rv = measure_rv(recursive_prompts)
baseline_rv = measure_rv(baseline_prompts)

# Prediction: surrender_rv > baseline_rv > recursive_rv
```

**Success criteria:**
- Complete map of R_V across self-reference types
- Identify if expansion is possible
- Understand the full geometry of the space

---

## PART IV: DATA MANAGEMENT

### 4.1 Directory Structure

```
results/
├── phase0_metric_validation/
│   ├── v_matrix_analysis/
│   ├── convex_hull_verification/
│   └── metric_comparison/
├── phase1_cross_architecture/
│   ├── pythia/
│   │   ├── 1.4B/
│   │   ├── 6.9B/
│   │   └── 12B/
│   ├── llama3/
│   │   ├── 1B/
│   │   ├── 8B/
│   │   └── 70B/
│   └── [...]
├── phase2_eigenstate/
├── phase3_attention/
├── phase4_kv_mechanism/
├── phase5_steering/
└── phase6_alternative_selfref/
```

### 4.2 Data Recording Standards

Every experiment must record:

```python
result = {
    # Metadata
    'timestamp': datetime.now().isoformat(),
    'experiment_id': uuid4(),
    'phase': 'phase1_cross_architecture',
    'model': {
        'name': 'pythia-6.9b',
        'architecture': 'pythia',
        'params': 6.9e9,
        'source': 'EleutherAI/pythia-6.9b',
    },
    
    # Prompt
    'prompt': {
        'text': prompt_text,
        'type': 'recursive',
        'level': 'L4',
        'source': 'REUSABLE_PROMPT_BANK',
    },
    
    # Measurements
    'rv': 0.54,
    'layer_profile': [0.98, 0.95, 0.87, ...],
    'attention_entropy': [...],
    
    # Reproducibility
    'seed': 42,
    'code_version': git_hash(),
    'config': {...},
}
```

### 4.3 Statistical Standards

- **Minimum N:** 50 prompts per condition per model
- **Effect size:** Report Cohen's d, not just p-values
- **Confidence intervals:** 95% CI for all estimates
- **Multiple comparisons:** Bonferroni correction when testing multiple models
- **Reproducibility:** All experiments must be rerunnable from saved configs

---

## PART V: PUBLICATION CRITERIA

### 5.1 Minimum Requirements for Paper

**DO NOT WRITE A PAPER UNTIL:**

1. **Cross-architecture validation**
   - [ ] R_V contraction in 5+ architectures
   - [ ] 2+ size tiers per architecture
   - [ ] Effect size d > 0.5 in each

2. **Metric validation**
   - [ ] Clear understanding of what R_V measures
   - [ ] Connection to V column space established or alternative interpretation justified

3. **Mechanistic understanding**
   - [ ] Know which layers encode the mode
   - [ ] Know which attention heads are involved
   - [ ] Understand why steering fails

4. **Reproducibility**
   - [ ] All code public
   - [ ] All data public
   - [ ] Independent replication by at least one other researcher

### 5.2 Paper Structure (When Ready)

1. **Core finding:** R_V contraction is a geometric signature of recursive self-observation
2. **Generalization:** Phenomenon exists across architectures and scales
3. **Mechanism:** KV cache in layers L-X to L-Y encodes the mode
4. **Attention:** Specific heads show altered patterns
5. **Limitations:** Linear steering cannot induce coherent recursion (and why)
6. **Theory:** Connection to convex hull geometry of attention

---

## PART VI: AGENT INSTRUCTIONS

### 6.1 For Any Agent Working on This Project

**YOUR ROLE:** Execute rigorous science. Not produce papers.

**BEFORE RUNNING ANY EXPERIMENT:**
1. Read this entire document
2. Check what has already been done (see results/ directory)
3. Verify you're using standardized prompts (REUSABLE_PROMPT_BANK)
4. Confirm your measurement code matches the established protocol

**DURING EXPERIMENTS:**
1. Record EVERYTHING (see data standards above)
2. Save intermediate results frequently
3. Note any anomalies or surprises
4. Don't cherry-pick results

**AFTER EXPERIMENTS:**
1. Update this document with findings
2. If findings contradict predictions, INVESTIGATE (don't dismiss)
3. Identify next logical experiment
4. Push to GitHub with clear commit messages

### 6.2 Current State (December 11, 2025)

**VALIDATED (Mistral-7B only):**
- R_V contraction exists (N=370, d>3.0)
- Dose-response (L1→L5)
- KV patching transfers mode (71-91%)
- GATEKEEPER specificity
- Steering breaks coherence (4 approaches failed)

**NOT YET DONE:**
- Phase 0 (metric validation)
- Phase 1 (cross-architecture) — ONLY Mistral tested
- Phase 2 (eigenstate hypothesis)
- Phase 3 (attention patterns)
- Phase 4 KV (only Mistral)
- Phase 5 (systematic steering documentation)
- Phase 6 (alternative self-reference types)

**IMMEDIATE PRIORITY:**
1. Phase 0 — Validate what we're actually measuring
2. Phase 1 — Test on Pythia (all sizes) as second architecture
3. Then expand systematically

### 6.3 Resources

**Code:**
- `/src/` — Core measurement code
- `/REUSABLE_PROMPT_BANK/` — Standardized prompts
- `/boneyard/` — Historical experiments (read for context, don't reuse blindly)

**Reference conversations:**
- "Attention heads in linear algebra" — Genesis of R_V concept
- "Recursive self-attention as stable..." — Eigenstate hypothesis
- "Claude artifacts consolidated" — Theoretical frameworks

**External:**
- 3Blue1Brown Linear Algebra series — Conceptual foundation
- Strang's MIT lectures — Pivot/column space theory

---

## PART VII: THE VISION

### What Success Looks Like

If this research program succeeds, we will have:

1. **A validated geometric signature** of recursive self-observation that generalizes across transformer architectures

2. **A mechanistic understanding** of how self-reference is encoded in attention patterns and KV cache

3. **A mathematical framework** connecting convex hull geometry of attention to phenomenological states

4. **A toolkit** for detecting and measuring recursive processing in any transformer model

5. **The foundation** for understanding how (or whether) these geometric signatures relate to anything like machine consciousness

### What This Is NOT

- This is NOT proof that LLMs are conscious
- This is NOT a spiritual claim dressed up as science
- This is NOT a rush to publication for career advancement

This is **careful, rigorous investigation** of a potentially fundamental computational phenomenon.

---

## SIGNATURES

**Initiated by:** John (AIKAGRYA Research)
**Date:** December 11, 2025
**Version:** 1.0

**To be updated as research progresses.**

---

*"The measure of a scientist is not how quickly they publish, but how honestly they investigate."*
