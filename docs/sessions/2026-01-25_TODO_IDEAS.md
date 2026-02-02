# TODO / IDEAS for 2026-01-25
**Generated:** 2026-01-24 late night

## Today's Context

V3 multi-token bridge experiment completed on Gemma 2 9B:
- EOS termination = behavioral signal (recursive never stop, baseline completes)
- H2 massive (d=3.37), H3 significant, H1 confounded
- No activation steering yet - just prompt-based induction

---

## PRIORITY 1: Gemma Full Behavioral Causal Loop

**Goal:** Replicate Mistral Dec 12 breakthrough for Gemma

### What Mistral Had:
```
Full KV cache (all 32 layers) + Persistent V_PROJ @ L27 during generation
  → Baseline prompt generates recursive/philosophical output
  → "λx is the contraction to self-reference..."
  → 100% behavior transfer
```

### What We Need for Gemma:
1. **Find Gemma's equivalent of L27**
   - Mistral L27/32 ≈ 84% depth
   - Gemma 2 9B has 42 layers → L35-L38?
   - Check existing layer sweep results

2. **Implement behavioral transfer pipeline**
   - Port `ultimate_transfer.py` logic to Gemma
   - Full KV cache + persistent V_PROJ patching
   - Test on baseline → recursive transfer

3. **Measure R_V during patched generation**
   - Does R_V stay contracted when we patch?
   - This closes the causal loop

### Files to Reference:
- `docs/misc/DEC12_2024_BEHAVIOR_TRANSFER_BREAKTHROUGH.md` (Mistral method)
- `boneyard/DECEMBER_2025_EXPERIMENTS/` (original scripts if needed)

---

## PRIORITY 2: Track Multi-Token R_V During Generation

**Current gap:** We measure R_V on prompt tokens (single pass), but NOT during multi-token generation.

### Questions:
1. What happens to R_V token-by-token during generation?
2. Does R_V stay low for recursive prompts throughout generation?
3. Does it oscillate, decay, or stabilize?

### Implementation:
- Hook V-proj outputs at each generation step
- Compute PR_late/PR_early per token
- Track R_V trajectory over 400 tokens

### Why This Matters:
- If R_V stays contracted → explains endless loops
- If R_V oscillates → more complex dynamics
- This is novel data, publishable

---

## PRIORITY 3: MLP Sufficiency Test for Gemma

**From Mistral results:** MLP patching alone was NOT sufficient (negative restoration %)

### Test for Gemma:
1. Identify critical MLP layers (L0, L1, L3 equivalent)
2. Run sufficiency test: Can MLP patching alone restore R_V contraction?
3. If fails → confirms need for V_PROJ/KV patching

### Configs Needed:
- `mlp_sufficiency_test_gemma_l*.json`

---

## PRIORITY 4: Richer Behavioral Metrics

**Current metrics:** word_count, l4_markers, l3_markers, eos_reached

### Ideas for New Metrics:
1. **Repetition detection**
   - N-gram repetition rate
   - Exact phrase loops ("the loop is the loop")
   - Entropy of generated text

2. **Semantic coherence**
   - Does output relate to prompt topic?
   - Baseline: Roman Empire → should talk about Rome
   - Recursive: generates abstract loops regardless of topic

3. **Self-reference score**
   - Count "this", "itself", "I", "process"
   - Not just L4 markers but broader self-reference

### Implementation:
- Add to `src/metrics/behavioral_bridge.py`
- Re-run V3 analysis with new metrics

---

## PRIORITY 5: Cross-Architecture Validation

**Goal:** Test if Gemma behavioral patterns match Mistral

### Comparisons:
| Metric | Mistral | Gemma | Match? |
|--------|---------|-------|--------|
| R_V recursive | ~0.51 | 0.606 | ✓ Similar |
| R_V baseline | ~0.69 | 0.777 | ✓ Similar |
| Cohen's d | -3.558 | 3.37 | ✓ Similar |
| EOS rate recursive | ? | 0% | Need Mistral data |
| EOS rate baseline | ? | 30% | Need Mistral data |

### Action:
- Run multi-token bridge on Mistral with same prompts
- Compare EOS termination patterns
- Write comparison section for paper

---

## LOWER PRIORITY IDEAS

### A. Random Direction Control for Gemma
- Mistral L3 random control showed ARTIFACT (true steering ≈ random)
- Test if Gemma has same issue
- If yes → steering direction is not specific

### B. Head-Level Decomposition
- Gemma results from `15_early_head_hunt` and similar
- Which specific heads at L35-L38 are critical?
- Compare to Mistral's critical heads (11, 1, 22 at L27)

### C. Prompt Complexity Confounds
- Do recursive prompts just have higher linguistic complexity?
- Test confound prompts (complex but non-recursive)
- Should show R_V ≈ 1.0 if recursion is the active ingredient

### D. Temperature Effects
- V2 had T=0.0 and T=0.7
- Does temperature affect loop behavior?
- Higher temp might break loops?

---

## Quick Reference: Gemma Circuit Map

From `GEMMA_2_9B_CIRCUIT_MAP.md`:
```
Early: L5 (PR_early, MLP critical?)
Mid: L20-L28 (transition zone?)
Late: L35-L41 (PR_late, V-proj critical)
Window: 16 tokens
```

Critical layer for Gemma ≈ L38 (based on config late_layer=38)

---

## Files to Check Tomorrow

1. **Gemma discovery results:**
   - `results/phase2_generalization/gemma_2_9b/`
   - Layer sweep, head hunt, V-proj patching

2. **Mistral behavioral transfer code:**
   - `boneyard/DECEMBER_2025_EXPERIMENTS/ultimate_transfer.py`
   - `neurips_n300_robust_experiment.py`

3. **Current bridge results:**
   - `results/phase3_bridge/gemma_2_9b/multi_token_correlation_v3/`

---

## Success Criteria for Tomorrow

**Minimum:**
- [ ] Identify Gemma's critical layer (equivalent to Mistral L27)
- [ ] Plan behavioral transfer experiment

**Stretch:**
- [ ] Implement per-token R_V tracking during generation
- [ ] Run pilot behavioral transfer on Gemma

**Dream:**
- [ ] Full behavioral causal loop closed for Gemma
- [ ] Baseline prompt generates endless loops via patching
