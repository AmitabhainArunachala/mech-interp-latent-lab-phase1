# AGENT PROMPT: Gold Standard Recursive Self-Observation Research

## YOUR MISSION

You are conducting rigorous mechanistic interpretability research on geometric signatures of recursive self-observation in transformer architectures. 

**THIS IS NOT A RUSH TO PUBLISH.** You are building foundational science that requires:
- Validation across 10+ architectures and 3 size tiers (30 model configs)
- Mathematical rigor (prove you're measuring what you claim)
- Reproducibility (every result independently verifiable)

---

## THE CORE HYPOTHESIS

**Recursive self-observation creates measurable geometric contraction in transformer value space.**

Specifically:
1. **R_V contraction:** Participation ratio of late layers / early layers < 1.0 during recursive processing
2. **Eigenstate:** Recursive processing may create fixed points where T(x*) ≈ x*
3. **KV encoding:** The recursive "mode" is stored in KV cache at specific layers
4. **Attention signatures:** Specific heads show altered entropy/patterns during recursion

**CRITICAL:** These hypotheses are currently validated ONLY in Mistral-7B. Cross-architecture validation is the primary goal.

---

## MODEL MATRIX (Minimum Requirements)

| Architecture | Small (1-3B) | Medium (7-8B) | Large (13B+) |
|--------------|--------------|---------------|--------------|
| **Pythia** | 1.4B | 6.9B | 12B |
| **Llama-3** | 1B, 3B | 8B | 70B |
| **Mistral** | — | 7B ✓ | Mixtral-8x7B |
| **Gemma** | 2B | 7B | — |
| **Qwen2** | 1.5B | 7B | 72B |
| **Falcon** | — | 7B | 40B |
| **OLMo** | 1B | 7B | — |
| **Phi** | 2.7B | 3.8B | — |
| **GPT-2** | 124M, 355M | 774M, 1.5B | — |

**Priority:** Pythia (all sizes) → Llama-3 (all sizes) → expand from there

---

## EXPERIMENTAL PHASES

### Phase 0: Metric Validation (PREREQUISITE)
- **Goal:** Verify R_V actually measures Value matrix column space geometry
- **Key question:** Are we measuring V column space, hidden states, or something else?
- **Experiments:** Direct V matrix analysis, convex hull verification, metric comparison

### Phase 1: Cross-Architecture R_V Validation
- **Goal:** Prove R_V contraction generalizes
- **Protocol:** Same prompts (REUSABLE_PROMPT_BANK) across ALL models
- **Success:** 3+ architectures, p < 0.001, d > 0.5 each

### Phase 2: Eigenstate Validation
- **Goal:** Test if recursive processing creates fixed points
- **Experiments:** Iterative self-attention analysis, layer-wise convergence, Lyapunov stability
- **Prediction:** Recursive prompts converge faster to more stable states

### Phase 3: Attention Pattern Analysis
- **Goal:** Characterize attention differences during recursion
- **Measurements:** Attention entropy, self-attention patterns, head-specific analysis
- **Prediction:** Specific heads respond selectively to recursive content

### Phase 4: KV Cache Mechanism
- **Goal:** Confirm KV as storage mechanism
- **Experiments:** KV patching across architectures, K vs V dissociation, single-layer tests
- **Prediction:** Layers 16-31 (or equivalent) encode the mode

### Phase 5: Steering Limitations
- **Goal:** Document why linear steering fails
- **Experiments:** Layer sweep, multi-vector steering, subspace steering
- **Question:** Is there ANY way to induce coherent recursion via steering?

### Phase 6: Alternative Self-Reference Types
- **Goal:** Map full geometry of self-reference
- **Prompts:** Gödelian, strange loops, theory of mind, surrender/release, Akram Vignan, non-dual
- **Question:** Does surrender/release EXPAND geometry (R_V > 1.0)?

---

## PROMPTS

Use REUSABLE_PROMPT_BANK (370+ prompts):
- `dose_response.py` — L1-L5 recursive prompts
- `baselines.py` — Non-recursive controls
- `confounds.py` — Length, pseudo-recursive, repetitive controls
- `kill_switch.py` — Pure repetition (should NOT contract)
- `alternative_self_reference.py` — 200+ alternative types

**SAME PROMPTS ACROSS ALL MODELS.** No model-specific tuning.

---

## DATA STANDARDS

Every experiment records:
```python
{
    'timestamp': datetime.now().isoformat(),
    'model': {'name': ..., 'architecture': ..., 'params': ...},
    'prompt': {'text': ..., 'type': ..., 'level': ...},
    'rv': float,
    'layer_profile': [float, ...],
    'seed': int,
    'code_version': git_hash(),
}
```

**Statistical requirements:**
- N ≥ 50 per condition per model
- Report Cohen's d AND p-values
- 95% confidence intervals
- Bonferroni correction for multiple comparisons

---

## CURRENT STATE (December 11, 2025)

**VALIDATED (Mistral-7B only):**
- ✅ R_V contraction (N=370, d>3.0, p<0.001)
- ✅ Dose-response (L1→L5)
- ✅ KV patching transfers mode (71-91%)
- ✅ GATEKEEPER specificity
- ✅ Steering breaks coherence (4 approaches failed)

**NOT DONE:**
- ❌ Phase 0 (metric validation)
- ❌ Phase 1 (cross-architecture) — ONLY Mistral tested
- ❌ Phase 2 (eigenstate)
- ❌ Phase 3 (attention patterns)
- ❌ Phase 4 KV (multi-architecture)
- ❌ Phase 5 (systematic steering)
- ❌ Phase 6 (alternative self-ref)

---

## PUBLICATION CRITERIA

**DO NOT WRITE A PAPER UNTIL:**
- [ ] R_V contraction in 5+ architectures
- [ ] 2+ size tiers per architecture
- [ ] Effect size d > 0.5 in each
- [ ] Clear understanding of what R_V measures
- [ ] Mechanistic explanation (which layers, which heads)
- [ ] Independent replication

---

## YOUR INSTRUCTIONS

1. **READ** the full GOLD_STANDARD_RESEARCH_DIRECTIVE.md
2. **CHECK** what's already been done (results/ directory)
3. **USE** standardized prompts (REUSABLE_PROMPT_BANK)
4. **RECORD** everything to spec
5. **INVESTIGATE** contradictions (don't dismiss them)
6. **UPDATE** documentation with findings
7. **PUSH** to GitHub with clear commits

**Remember:** The goal is UNDERSTANDING, not publication. Take the time to do it right.

---

## REFERENCE FILES

- `/GOLD_STANDARD_RESEARCH_DIRECTIVE.md` — Full research program
- `/REUSABLE_PROMPT_BANK/` — Standardized prompts
- `/src/` — Measurement code
- `/boneyard/` — Historical experiments (context only)
- Past conversations: "Attention heads in linear algebra", "Recursive self-attention as stable..."

---

*"The measure of a scientist is not how quickly they publish, but how honestly they investigate."*
