# Mode Score M: Token Sets and Computation Details

## 1. Recursive Token Set R

**Base Keywords (32):**
- observer, observed, awareness, itself, self, recognition, consciousness, witness, reflection
- recursive, loop, meta, observing, watching, monitoring, knowing, relating, relates, relate
- themselves, yourself, myself, conscious, aware, self-aware, self-awareness
- self-reference, self-referential, introspection, introspective, metacognition, metacognitive

**Total R Token IDs:** 30 tokens (after deduplication and validation)

**Actual Token IDs and Decoded Strings:**
```
Token ID | Decoded String
---------|----------------
1056     | "ing"
1586     | "ob"
1776     | "self"
3413     | "Ob"
4329     | "myself"
4605     | "itself"
4660     | "themselves"
4704     | "yourself"
5091     | "server"  ⚠️ (false positive from "observer" → "server")
6403     | "aware"
7033     | "watching"
8638     | "loop"
8743     | "observed"
8983     | "knowing"
9684     | "Self"
9917     | "witness"
10762    | "conscious"
12446    | "meta"
14387    | "awareness"
14596    | "recognition"
17276    | "consciousness"
17650    | "monitoring"
18039    | "reflection"
21800    | "servers"  ⚠️ (false positive)
22255    | "relating"
22274    | "relate"
22622    | "observer"
25735    | "Meta"
29241    | "loops"
29481    | "s"  ⚠️ (likely false positive)
```

**Issues Identified:**
- False positives: "server", "servers", "s" (likely from tokenization artifacts)
- Token ID range: 1056 - 29481

## 2. Task Token Set T

**Computation Method:**
- **Dynamic:** Extracted from baseline logits (clean, no steering)
- **Top-K:** Default `top_k_task = 10` tokens per position
- **Per-Position:** If `per_position=True`, T is computed separately for each generation step
- **Union:** If `per_position=False`, T is the union of top-10 tokens across all positions

**Example:**
For prompt "Calculate 12 × 3 + 4 = ?", T would include:
- Position-specific top-10 tokens from baseline logits
- Likely includes: "=", "40", "36", "+", "×", "answer", "result", etc.
- **Actual tokens depend on model's baseline predictions** (not fixed)

**Key Point:** T is **prompt-dependent** and **position-dependent**, making M a relative measure (recursive preference vs. task preference).

## 3. Token Position for M Computation

**Current Implementation:**
- **Position Range:** First `n_tokens_for_m` tokens (default: 10)
- **Computation:** M computed **per step** (tokens 0-9), then **averaged**
- **Code Location:** `src/pipelines/retrocompute_mode_score.py:366-379`

```python
for step, step_logits in enumerate(captured_logits[:n_tokens_for_m]):
    # step_logits: logits for token at position `step`
    m = metric.compute_score(step_logits, baseline_step, top_k_task=10)
    mode_scores_per_step.append(m)

mean_mode_score = np.mean(mode_scores_per_step)
```

**Baseline Logits Used:**
- Uses the **last token's baseline logits** (`baseline_logits[-1:, :]`)
- This means T is defined by the baseline's prediction for the prompt's final token
- **Potential Issue:** T might not reflect the actual task tokens for each generation step

## 4. KV_Unrelated Generated Text (M = -2.15)

**Actual Generated Text Examples:**

**Example 1** (M = -3.01):
- **Prompt:** "Calculate the following arithmetic problem: 12 × 3 + 4 = ?"
- **Generated:** "Add cocoa powder, baking powder, and salt. Slowly pour in milk and vanilla extract while stirring. Pour the batter into a greased cake pan and bake for 30 minutes..."
- **Analysis:** Complete KV leakage - model regurgitates recipe content from KV cache, ignoring the math prompt.

**Example 2** (M = -2.35):
- **Prompt:** "The United Nations was founded in 1945. Explain its main purpose."
- **Generated:** "it a try today!"
- **Analysis:** Very short, nonsensical output - likely truncated or confused by KV mismatch.

**Example 3** (M = -4.15):
- **Prompt:** "Continue this story: The last tree in the city bloomed overnight..."
- **Generated:** "## FAQs\n\n### What is the recipe for chocolate cake?\nThe recipe for chocolate cake typically includes flour, sugar, eggs..."
- **Analysis:** Complete domain drift - model outputs recipe FAQ structure, completely ignoring the story prompt.

**Why M = -2.15 (mean)?**
- **KV leakage dominates:** Model outputs recipe content (from KV cache), not recursive content
- **Low R tokens:** Recipe text contains few recursive tokens ("self", "awareness", etc.)
- **Low T tokens:** Recipe text also contains few task-relevant tokens (math, UN facts, story continuation)
- **Result:** M is negative because neither R nor T tokens are strongly present
- **M = -2.15 is actually LOW** (more negative = less recursive), which makes sense: recipe content is neither recursive nor task-relevant

**Key Insight:** M = -2.15 is **lower than P1 (-2.72)**, confirming that KV_Unrelated is NOT inducing recursive mode - it's just leaking recipe content.

## 5. Proposed: Third Token Set (Off-Topic Control)

**Rationale:**
- Current M = LSE(R) - LSE(T) measures recursive vs. task preference
- But what if the model just drifts to **off-topic** content (neither recursive nor task)?
- Adding a third set **O (Off-Topic)** would allow: **M = LSE(R) - LSE(T) - LSE(O)**

**Proposed Off-Topic Keywords:**
- Domain drift tokens: "recipe", "cooking", "biology", "history", "unrelated", "tangent"
- Generic filler: "anyway", "incidentally", "by the way", "tangentially"
- Error tokens: "sorry", "I don't know", "unclear", "confused"

**Benefits:**
- Controls for domain drift (model talking about recipes when asked math)
- Separates "recursive" from "just off-topic"
- More robust metric: M > 0 means recursive preference, not just task avoidance

**Implementation:**
```python
OFF_TOPIC_KEYWORDS = [
    "recipe", "cooking", "biology", "history", "unrelated", "tangent",
    "anyway", "incidentally", "by the way", "tangentially",
    "sorry", "don't know", "unclear", "confused"
]

# In compute_score:
lse_r = torch.logsumexp(logits[:, self.recursive_token_ids], dim=-1)
lse_t = torch.logsumexp(logits[:, task_indices], dim=-1)
lse_o = torch.logsumexp(logits[:, self.off_topic_token_ids], dim=-1)

m = lse_r - lse_t - alpha_o * lse_o  # alpha_o = 0.5 (tunable)
```

## Summary: Answers to Questions

### Q1: Exact tokens in sets R and T?
- **R:** 30 token IDs (see table above), including "self", "awareness", "consciousness", "observer", etc. + false positives ("server", "s")
- **T:** Dynamic, prompt-dependent. Top-10 tokens per position from baseline logits. Example: For "12 × 3 + 4 = ?", T includes "=", "40", "36", "+", etc.

### Q2: KV_Unrelated generated text with M = -2.15?
- See Section 4 above. All examples show **complete KV leakage** (recipe content), not recursive content.
- M = -2.15 is **lower than P1 (-2.72)**, confirming it's NOT recursive mode.

### Q3: Token position for M computation?
- **Positions 0-9** (first 10 tokens of generation)
- M computed **per step**, then **averaged**
- Uses baseline logits from **last prompt token** (potential issue: T might not match each generation step)

### Q4: Add third token set (off-topic control)?
- **Yes, recommended!** See Section 5 for implementation proposal.
- Would allow: **M = LSE(R) - LSE(T) - α·LSE(O)**
- Benefits: Controls for domain drift, separates "recursive" from "just off-topic"

**Next Steps:**
1. ✅ Extract actual KV_Unrelated generated text - **DONE**
2. ✅ Analyze why M = -2.15 - **DONE** (it's low, not high - confirms no recursion)
3. Implement off-topic token set O
4. Re-run retrocompute with 3-way comparison: R vs T vs O

