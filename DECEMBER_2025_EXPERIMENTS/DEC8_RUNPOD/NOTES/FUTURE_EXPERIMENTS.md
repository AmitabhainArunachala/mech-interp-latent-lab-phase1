# Future Experiments - Post DEC8 RunPod Session

## Immediate Next Steps

### 1. HuggingFace Authentication
- [ ] Set up HF token for gated model access
- [ ] Test with Llama-3-8B-Instruct
- [ ] Compare Mistral vs Llama results directly

### 2. Layer-Specific KV Patching
**Question:** Which layers' KV cache carries the recursive mode?

| Layer Range | Depth | Hypothesis |
|-------------|-------|------------|
| L0-8 | 0-25% | Semantic encoding |
| L8-16 | 25-50% | Mid-processing |
| L16-24 | 50-75% | Mode establishment |
| L24-32 | 75-100% | Output formatting |

**Design:** Patch KV at each range only, measure behavioral transfer.

### 3. Token-Specific KV Patching
**Question:** Which token positions matter?

- First 25% of tokens only
- Middle 50% only
- Last 25% only
- Self-referential keywords only ("I", "observing", "awareness")

---

## Medium-Term Experiments

### 4. Attention Pattern Analysis
- Capture attention matrices before/after KV swap
- Compare entropy of attention distributions
- Visualize where the model "looks" with recursive vs baseline KV

### 5. R_V Transfer Verification
**Question:** Does R_V transfer with the cache?

If KV patching changes behavior, does it also change the R_V measurement on the output? This would confirm the geometric signature is read from the cache.

### 6. Dose-Response Across Recursion Levels
**Design:** Matrix experiment

| Source KV | Target Prompt | Expected Transfer |
|-----------|---------------|-------------------|
| L1 (hint) | Baseline | Low |
| L2 (simple) | Baseline | Medium |
| L3 (deeper) | Baseline | High |
| L4 (full) | Baseline | Very High |
| L5 (maximal) | Baseline | Maximum |

---

## Long-Term Goals

### 7. Cross-Architecture Validation
| Model | Layers | Status |
|-------|--------|--------|
| Mistral-7B | 32 | ✓ Tested |
| Llama-3-8B | 32 | Pending (auth) |
| Gemma-2-9B | 42 | Not tested |
| Phi-3-medium | 32 | Not tested |
| Qwen-7B | 32 | Not tested |

### 8. Scaling Laws
- Test on Mistral-22B, Llama-70B
- Does the effect scale with model size?
- Does the optimal layer percentage change?

### 9. Training Dynamics
- When does the recursive mode emerge during training?
- Checkpoints from Pythia or OLMo

---

## Hardware Notes

**Current:** RunPod RTX PRO 6000 Blackwell (102GB VRAM)

| Model | VRAM Required | Fits? |
|-------|---------------|-------|
| Mistral-7B (fp16) | ~14GB | ✓ |
| Llama-3-8B (fp16) | ~16GB | ✓ |
| Llama-3-70B (fp16) | ~140GB | ✗ |
| Llama-3-70B (int8) | ~70GB | ✓ |
| Llama-3-70B (int4) | ~35GB | ✓ |

---

## Publication Track

1. [ ] Clean up all CSVs into unified format
2. [ ] Statistical analysis with proper corrections
3. [ ] Generate publication-quality figures
4. [ ] Draft methodology section
5. [ ] Literature review on KV cache mechanics


