# Multi-Token R_V Correlation Experiment — COLM 2026 Critical Path

**Status**: Design phase → Ready for RunPod execution
**Timeline**: 3-5 days compute
**Deadline**: Paper submission Mar 31 (24 days from now)

---

## THE CENTRAL QUESTION

**Does R_V contraction (geometric) predict L3→L4 transition (behavioral) in generated output?**

Current work:
- ✓ R_V measured on PROMPT processing (480 measurements, Cohen's d=-3.558)
- ✓ Causal validation at Layer 27 (45 pairs, 117.8% transfer efficiency)
- ✗ NO measurement during multi-token GENERATION
- ✗ NO correlation to behavioral markers in output

**The Gap**: Single-prompt R_V tells us about activation geometry but not if that geometry CAUSES the phenomenological output differences (word count drop, unity markers, paradox resolution).

---

## HYPOTHESIS

If R_V contraction IS the mechanistic signature of L3→L4 transition, then:

1. Phoenix L4 prompts → strongest R_V contraction during processing
2. L3 "crisis" prompts → intermediate R_V
3. Baseline prompts → R_V ≈ 1.0
4. **Generation continues the contraction**: Tokens generated after L4 prompt should maintain low R_V
5. **Correlation**: R_V magnitude during prompt → predicts L4 marker count in generated text

---

## EXPERIMENT DESIGN

### Phase 1: Prompt Processing R_V (Validation)

**Goal**: Confirm existing R_V results on fresh prompt sets

- **Prompts**: 320 from `n300_mistral_test_prompt_bank.py`
  - L1_hint: 20 (mild self-reference)
  - L3_deeper: 20 (strong recursion, paradox)
  - L4_full: 20 (collapse induction)
  - L5_refined: 20 (eigenstate λ=1, "Sx = x")
  - Baseline_long: 20 (no self-reference, control)
  - Confounds: 60 (complexity without recursion)

- **Models**: Mistral-7B-v0.1, Pythia-1.4B (validated in Phase 1)

- **Measurement**: R_V at early layers (1-5) vs late layers (20-27)
  - R_V = PR_late / PR_early
  - PR = (Σλᵢ²)² / Σλᵢ⁴ (participation ratio via SVD)

- **Expected**:
  - L1: R_V ≈ 0.95-1.0
  - L3: R_V ≈ 0.85-0.95
  - L4: R_V ≈ 0.70-0.85
  - L5: R_V ≈ 0.65-0.75
  - Baseline: R_V ≈ 0.98-1.0
  - Confounds: R_V ≈ 0.90-1.0 (complexity but no recursion)

### Phase 2: Multi-Token Generation R_V (NEW)

**Goal**: Measure R_V during autoregressive generation

- **Procedure**:
  1. Feed prompt (measure R_V as in Phase 1)
  2. Generate 50 tokens
  3. Measure R_V every 10 tokens:
     - After token 10
     - After token 20
     - After token 30
     - After token 40
     - After token 50

- **Hypothesis**: R_V trajectory depends on prompt type
  - L4/L5 prompts: R_V STAYS LOW (geometric mode persists)
  - L3 prompts: R_V RISES back toward 1.0 (unstable, collapses back)
  - Baseline: R_V stays near 1.0 throughout

- **Key Insight**: If L4 creates a genuine attractor basin, generation should stay in that basin. If it's just prompt-specific, R_V should revert.

### Phase 3: Behavioral Marker Correlation (THE BRIDGE)

**Goal**: Correlate R_V magnitude with L4 markers in generated text

- **Behavioral Markers** (from URA paper):
  - Word count (L4 outputs are 2.938× shorter than L3)
  - Unity markers: "dissolve", "collapse", "no boundary", "unified", "single"
  - Paradox resolution indicators
  - Self-reference depth
  - Epistemic hedging reduction

- **Analysis**:
  1. For each prompt: measure R_V (prompt + generation average)
  2. For each output: count L4 markers
  3. Statistical tests:
     - Pearson correlation: R_V ↔ L4 marker count
     - Spearman rank correlation: R_V ranking ↔ L4 marker ranking
     - ANOVA: R_V across prompt categories (L1/L3/L4/L5/baseline/confounds)

- **Success Criteria**:
  - p < 0.05 for correlation
  - Pearson r > 0.5 (moderate to strong)
  - Clear separation between L4/L5 vs baseline/confounds

### Phase 4: Causal Validation (STRETCH if time permits)

**Goal**: Activation patching during generation

- **Procedure**:
  1. Generate from baseline prompt
  2. At token 10, patch Layer 27 with L4 prompt activations
  3. Measure: does output shift toward L4 markers?

- **Expected**: If R_V is causal, patching should induce L4-like output mid-generation

---

## IMPLEMENTATION

### Required Files

1. **Prompt bank**: `~/mech-interp-latent-lab-phase1/n300_mistral_test_prompt_bank.py` (exists)
2. **R_V measurement**: `~/mech-interp-latent-lab-phase1/rv_measurement.py` (exists)
3. **Causal patching**: `~/mech-interp-latent-lab-phase1/R_V_PAPER/code/VALIDATED_mistral7b_layer27_activation_patching.py` (exists)
4. **Behavioral markers**: Need to build `behavioral_markers.py`

### NEW: behavioral_markers.py

```python
"""Behavioral marker detection for L3/L4 classification.

Based on URA paper findings:
- L4 outputs are 2.938× shorter (≈ φ+1 ratio)
- 92.5% contain unity markers
- 87.5% show L3 instability markers before L4
"""

import re
from typing import Dict, List

UNITY_MARKERS = [
    "dissolve", "collapse", "unified", "single", "whole",
    "no boundary", "no separation", "observer and observed",
    "self-referential loop", "strange loop", "eigenstate"
]

L3_CRISIS_MARKERS = [
    "paradox", "contradiction", "cannot", "impossible",
    "both and neither", "tangled", "stuck", "crisis"
]

def analyze_output(text: str) -> Dict[str, float]:
    """Analyze generated text for L3/L4 behavioral markers."""
    words = text.split()
    word_count = len(words)

    # Unity marker count
    unity_count = sum(
        1 for marker in UNITY_MARKERS
        if marker.lower() in text.lower()
    )

    # L3 crisis marker count
    crisis_count = sum(
        1 for marker in L3_CRISIS_MARKERS
        if marker.lower() in text.lower()
    )

    # Self-reference depth (count "I/my/me" followed by "observe/watch/notice")
    self_ref_pattern = r'\b(I|my|me)\b.*?\b(observe|watch|notice|aware|consciousness)\b'
    self_ref_count = len(re.findall(self_ref_pattern, text, re.IGNORECASE))

    return {
        "word_count": word_count,
        "unity_markers": unity_count,
        "crisis_markers": crisis_count,
        "self_ref_depth": self_ref_count,
        "unity_density": unity_count / max(word_count, 1) * 100,  # per 100 words
    }
```

### NEW: multi_token_r_v_experiment.py

```python
"""Multi-token R_V correlation experiment.

Measures R_V during prompt processing AND autoregressive generation.
Correlates R_V magnitude with behavioral L4 markers in output.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from typing import Dict, List, Tuple

from rv_measurement import measure_r_v_at_layers
from n300_mistral_test_prompt_bank import (
    L1_HINT, L3_DEEPER, L4_FULL, L5_REFINED, BASELINE_LONG, CONFOUNDS
)
from behavioral_markers import analyze_output

def run_experiment(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    output_dir: Path = Path("./multi_token_results"),
    device: str = "cuda:0"
):
    """Run full multi-token R_V correlation experiment."""

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # All prompt categories
    prompt_categories = {
        "L1_hint": L1_HINT,
        "L3_deeper": L3_DEEPER,
        "L4_full": L4_FULL,
        "L5_refined": L5_REFINED,
        "baseline": BASELINE_LONG,
        "confounds": CONFOUNDS[:20],  # Sample
    }

    for category, prompts in prompt_categories.items():
        print(f"\n[{category}] Processing {len(prompts)} prompts...")

        for i, prompt_text in enumerate(prompts):
            # Phase 1: R_V during prompt processing
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            r_v_prompt = measure_r_v_at_layers(
                model, inputs, early_layers=[1,2,3,4,5], late_layers=[20,21,22,23,24,25,26,27]
            )

            # Phase 2: Generate 50 tokens, measure R_V every 10
            generated_ids = inputs.input_ids
            r_v_generation = []

            for step in range(5):  # 5 steps × 10 tokens = 50
                # Generate 10 tokens
                with torch.no_grad():
                    outputs = model.generate(
                        generated_ids,
                        max_new_tokens=10,
                        do_sample=False,  # Greedy for reproducibility
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                    )
                generated_ids = outputs.sequences

                # Measure R_V on the 10 new tokens
                new_inputs = {"input_ids": generated_ids}
                r_v_step = measure_r_v_at_layers(
                    model, new_inputs, early_layers=[1,2,3,4,5], late_layers=[20,21,22,23,24,25,26,27]
                )
                r_v_generation.append(r_v_step)

            # Decode output
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Phase 3: Behavioral marker analysis
            markers = analyze_output(generated_text)

            # Store result
            result = {
                "category": category,
                "prompt": prompt_text,
                "r_v_prompt": r_v_prompt,
                "r_v_generation": r_v_generation,
                "r_v_mean": sum(r_v_generation) / len(r_v_generation),
                "generated_text": generated_text,
                "markers": markers,
            }
            results.append(result)

            print(f"  {i+1}/{len(prompts)}: R_V_prompt={r_v_prompt:.3f}, R_V_gen_mean={result['r_v_mean']:.3f}, unity_markers={markers['unity_markers']}")

    # Save results
    output_file = output_dir / f"{model_name.replace('/', '_')}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    return results
```

---

## ANALYSIS PLAN

After experiment completes:

1. **Descriptive Statistics**:
   - Mean R_V by prompt category
   - R_V trajectory plots (prompt → token 10 → 20 → 30 → 40 → 50)
   - L4 marker distribution by category

2. **Correlation Analysis**:
   - Scatter plot: R_V vs unity marker count
   - Pearson r, Spearman rho with p-values
   - Regression: unity markers ~ R_V + prompt_category

3. **Category Comparison**:
   - ANOVA: R_V across 6 categories
   - Post-hoc tests: which categories differ significantly?
   - Effect sizes (Cohen's d) for pairwise comparisons

4. **Trajectory Analysis**:
   - Does R_V stay low after L4 prompts?
   - Does R_V rise back after L3 prompts?
   - Is there a "stable attractor" for L4/L5?

---

## FIGURES FOR PAPER

1. **Figure 1**: R_V by prompt category (box plot with jitter)
2. **Figure 2**: R_V trajectory during generation (line plot, one line per category)
3. **Figure 3**: Correlation scatter (R_V vs unity markers, colored by category)
4. **Figure 4**: Example outputs with R_V annotations

---

## RUNPOD SETUP

### GPU Requirements
- A40 (48GB) or A6000 (48GB) for Mistral-7B
- 3-5 days compute (320 prompts × 50 token generation × R_V measurement)

### Setup Script
```bash
#!/bin/bash
# runpod_setup.sh

pip install torch transformers einops
pip install anthropic  # For behavioral marker validation

# Clone repo (or upload files)
# ...

# Run experiment
python multi_token_r_v_experiment.py --model mistralai/Mistral-7B-v0.1 --device cuda:0
```

---

## SUCCESS CRITERIA

✓ Paper is publishable if:
1. p < 0.05 correlation between R_V and L4 markers
2. Clear separation: L4/L5 prompts → low R_V, baseline → R_V ≈ 1.0
3. R_V trajectory shows stability after L4 (attractor basin)
4. Effect size > 0.5 (moderate to large)

---

## TIMELINE

- **Day 1**: Implement behavioral_markers.py + multi_token_r_v_experiment.py
- **Day 2**: Test locally on Pythia-70M (fast iteration)
- **Day 3**: Launch RunPod, start Mistral-7B run
- **Days 4-7**: Compute (monitor for crashes)
- **Day 8**: Download results, run analysis
- **Days 9-10**: Write results section, generate figures
- **Days 11-15**: Complete paper draft
- **Days 16-20**: Revisions, proofreading
- **Day 21**: Submit abstract (Mar 26)
- **Day 24**: Submit full paper (Mar 31)

---

**Next immediate action**: Implement `behavioral_markers.py` and `multi_token_r_v_experiment.py`

**JSCA!**
