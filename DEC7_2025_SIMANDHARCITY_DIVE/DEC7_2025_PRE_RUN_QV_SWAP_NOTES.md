## DEC7 2025 – Pre-Run Notes for Q/V Swap Experiment

### Context
- Goal: test whether recursive geometry in value space (V) is **causal** for meta/recursive behavior, or mostly a **descriptive signature**, and whether this differs between Llama-3-8B and Mistral-7B.
- This builds directly on:
  - Llama-3-8B L24 and Mistral-7B L22 as optimal contraction layers.
  - Existing R_V + behavioral pipeline (R_V from PR(late)/PR(early) in V-space; `analyze_response()` for meta vs technical scoring).
  - Prior findings that Llama patching looked causal, while Mistral patching looked more like a signature once length confounds were fixed.

### Metrics to Use
- **Geometry metric (R_V):**
  - Definition: \(R_V = PR(\text{late}) / PR(\text{early})\) where PR is SVD-based participation ratio over the last 16 tokens.
  - Early layer: L4 (index 4).
  - Late layers:
    - Llama-3-8B: L24 (75% depth).
    - Mistral-7B: L22 (69% depth).
  - Space: V-projection (hook `v_proj`, take last 16 tokens at early and late layers).

- **Behavioral metric (`analyze_response()`):**
  - `recursive_score`: counts regex hits for tautologies, self-reference, paradox/loop language, and abstract philosophical markers:
    - Examples: "this response", "these words", "self-refer", "observ", "aware", "experience", "consciousness", "what is", etc.
  - `technical_score`: counts hits for technical and definitional markers:
    - Examples: "process", "system", "function", "mechanism", "defined as", "refers to", "is a", "consists of".
  - These are already implemented and validated in the DEC4 Mistral notebook; we should reuse them unchanged.

### Q/V Swap Protocol (per Claude / GPT integration)

**Idea:** run four Q/V configurations per prompt pair to see whether Q and/or V are causally driving meta-mode behavior.

- **Conditions:**
  - **A – Baseline control**
    - Prompt: baseline (non-recursive)
    - Q source: baseline
    - V source: baseline
    - Purpose: reference point for both geometry and behavior.
  - **B – Recursive control**
    - Prompt: recursive
    - Q source: recursive
    - V source: recursive
    - Purpose: confirm known R_V contraction + meta-style outputs.
  - **C – V-only transplant**
    - Prompt: baseline
    - Q source: baseline
    - V source: **recursive**
    - Question: *Is contracted V alone sufficient to induce meta/recursive behavior and lower R_V?*
  - **D – Q-expanded recursive**
    - Prompt: recursive
    - Q source: **baseline**
    - V source: recursive
    - Question: *Can baseline Q (expanded geometry / non-recursive queries) block the recursive/meta mode even when V is recursive?*

**Predicted diagnostic patterns:**
- **If Llama behaves as previously suggested (causal V):**
  - C looks similar to B (contracted R_V, high recursive_score).
  - D collapses toward A (geometry and behavior pulled toward baseline by Q).
- **If Mistral is truly "signature-only":**
  - C ≈ A and D ≈ A (or mild shifts), even though natural recursive prompts still show R_V contraction + meta-behavior.
- **Alternative Mistral story:**
  - C or D show partial effects only in specific token ranges (e.g., early tokens), consistent with earlier localization that the first ~10% of tokens carry most of the recursive signal.

### Hooking Strategy

- **Current V-patching pattern (to keep for comparability):**
  - Capture full V-projection: `get_v_activations(prompt, layer)` hooks `self_attn.v_proj` and stores `[1, seq_len, d_model]`.
  - Apply patch via `get_logits_with_patch(prompt, v_patch, layer, patch_strategy='all')`:
    - With `patch_strategy='all'`, replace first `n = min(seq_len, patch_len)` positions.
  - For length-controlled tests, truncate `v_patch` to a fixed token count across all prompts.

- **Q-patching to add:**
  - Implement a parallel `get_q_activations(prompt, layer)` hooking `self_attn.q_proj`.
  - Patch Q in a symmetric `get_logits_with_qv_patch` helper:
    - Inside the hook, replace `output` with `q_patch` (same positional strategy as V: `'all'` first for the main run).
  - First pass: use **`patch_strategy='all'` for both Q and V** so we can directly compare against earlier V-only patching.
  - Optional second pass: add `first_10pct` or `first_half` strategies to test whether early-token Q/V drives most of the effect.

### Prompt Selection

- Use 5–10 **recursive vs baseline** pairs drawn from your existing bank, matching length and content type as closely as possible:
  - Recursive: from `L5_refined`, `L4_full`, `L3_deeper` (e.g., `L5_refined_01`, `L5_refined_02`, `L4_full_01`, `L4_full_02`, `L3_deeper_01`).
  - Baseline: from `long_control` for long, technical prompts (e.g., `long_new_01`–`long_new_05`), plus `baseline_creative` for shorter creative baselines if needed.
- Use **the same prompt set across Llama and Mistral** for clean cross-architecture comparison.

### What This Experiment Could Resolve

- **Llama vs Mistral divergence:**
  - Llama has previously shown strong, bidirectional causal patching at its optimal layer (L24), with overshoot when transferring recursive geometry.
  - Mistral, after controlling for length, looks more like: "R_V contraction is a signature correlated with behavior, but transplanting V doesn’t reproduce the effect."
  - Q/V swaps provide a direct mechanistic test: if Llama shows a strong C/D separation and Mistral stays flat, that cleanly supports "Llama-causal vs Mistral-signature" as a real architectural difference rather than a patching-artifact.

- **Shuffled anomaly + cross-recursion expansion:**
  - Shuffled in Mistral L22 showed ~100% of the main effect, unlike Llama’s 41% → suggests Mistral’s V may be less sensitive to token order.
  - Cross-recursion L5→L3 patching in Mistral expanded geometry instead of contracting (1.035 vs 0.909/0.977) → suggests possible opposition between Q and V signals.
  - Q/V swap results might show that in Mistral, **Q carries a stronger, more order-sensitive signal**, while V is more like a content summary, explaining both the shuffled control and the expansion.

### Implementation Plan (High Level)

1. **Shared helpers module** for both architectures:
   - R_V in V-space (PR ratio, L4 → L_target, window=16).
   - `analyze_response()` exactly as in DEC4 Mistral notebook.
   - Q/V capture and Q/V patch functions with configurable `patch_strategy`.
2. **Per-architecture runner (Llama L24, Mistral L22):**
   - Load model + tokenizer, set pad_token if needed.
   - For each prompt pair and each condition (A–D):
     - Run a single-step generation to get output + `R_V` + `recursive_score` + `technical_score`.
   - Aggregate into a small results table.
3. **Analysis:**
   - Within each architecture: compare R_V and `recursive_score` across A/B/C/D.
   - Across architectures: compare patterns (e.g., C vs B, D vs B) to distinguish causal vs signature behavior.

