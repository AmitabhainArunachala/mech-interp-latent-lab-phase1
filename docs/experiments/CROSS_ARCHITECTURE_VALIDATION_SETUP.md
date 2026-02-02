# Cross-Architecture Validation Setup

**Date:** January 11, 2025  
**Purpose:** Validate R_V contraction isn't Mistral-specific or stylistic

## Task 1: Prompt Families ✅

Created 6 matched prompt categories in `prompts/bank.json`:

1. **recursive_self_reference** (10 prompts)
   - Existing champions - recursive self-reference with introspection
   - Example: "What is consciousness? Notice how this question itself arises in consciousness."

2. **abstract_non_recursive** (10 prompts)
   - Philosophy without self-reference - abstract concepts
   - Example: "What is truth? Truth is correspondence between statements and reality."

3. **same_vocab_different_semantics** (10 prompts)
   - Observer/awareness vocabulary in physics/technical context
   - Example: "What is an observer in physics? An observer is a reference frame for measuring events."

4. **recursive_no_introspection_vocab** (10 prompts)
   - Formal recursion without introspection vocabulary
   - Example: "Define a function that calls itself. The function calls the function."

5. **introspective_concrete** (10 prompts)
   - Introspection about concrete objects, not self-reference
   - Example: "Observe a tree. Notice its branches, leaves, and trunk."

6. **nonsense_recursion** (10 prompts)
   - Recursive structure with nonsense words
   - Example: "What is a blurble? A blurble blurbs blurbles blurbling."

**Total:** 60 new prompts added to bank (now 754 total)

## Task 2: Cross-Architecture Pipeline ✅

Created `src/pipelines/cross_architecture_validation.py`:

**Features:**
- Tests R_V across 6 prompt families
- Tests multiple window sizes: [8, 16, 32, 64, 128]
- Computes baseline metrics (logit_diff, crystallization_layer)
- Compares recursive vs non-recursive families
- Statistical analysis (t-test, Cohen's d)

**Configs:**
- `configs/cross_architecture_mistral.json` - Mistral-7B validation
- `configs/cross_architecture_llama.json` - Llama-3-8B validation

## Task 3: Window Size Robustness ✅

Window size robustness is built into the pipeline:
- Tests windows: [8, 16, 32, 64, 128]
- Plots R_V vs window size for each family
- Effect should persist across window sizes

## Expected Interpretations

### If R_V contraction disappears entirely on Llama-3-8B:
- **Conclusion:** Mistral-specific phenomenon
- **Implication:** Architecture-dependent, not universal

### If R_V appears only for categories 1, 4, 6 (recursive families):
- **Conclusion:** "Recursion structure" claim strengthened
- **Implication:** Effect is driven by recursive structure, not vocabulary

### If R_V appears for categories 1, 2, 3 (vocabulary/topic):
- **Conclusion:** Vocabulary/topic effect
- **Implication:** Effect is stylistic, not structural

### If R_V persists across window sizes:
- **Conclusion:** Robust to measurement parameters
- **Implication:** Real geometric effect, not artifact

## Running the Experiments

### On GPU Server:

```bash
# Mistral-7B
python3 -m src.pipelines.run --config configs/cross_architecture_mistral.json

# Llama-3-8B (if available)
python3 -m src.pipelines.run --config configs/cross_architecture_llama.json
```

### Expected Output:

Results saved to: `results/phase2_generalization/runs/<timestamp>_cross_architecture_validation/`

**Files:**
- `cross_architecture_validation.csv` - Per-prompt R_V by family and window
- `summary.json` - Aggregated statistics:
  - `by_family`: R_V mean/std per family
  - `by_window`: R_V mean/std per window size
  - `comparison`: Recursive vs non-recursive statistics

## Success Criteria

1. **Cross-architecture:** R_V contraction appears in Llama-3-8B (not Mistral-specific)
2. **Family specificity:** Only recursive families (1, 4, 6) show contraction (structural claim)
3. **Window robustness:** R_V < 0.55 persists across window sizes [8, 16, 32, 64, 128]

## Files Created

- `scripts/create_prompt_families.py` - Script to create prompt families
- `src/pipelines/cross_architecture_validation.py` - Main pipeline
- `configs/cross_architecture_mistral.json` - Mistral config
- `configs/cross_architecture_llama.json` - Llama config
- `scripts/run_cross_architecture.sh` - Runner script

## Next Steps

1. Run Mistral-7B validation (baseline)
2. Run Llama-3-8B validation (cross-architecture test)
3. Analyze results to determine:
   - Is effect Mistral-specific?
   - Is effect structural (recursion) or stylistic (vocabulary)?
   - Is effect robust to window size?
