# Opus 4.1: Experiment 1 - Behavioral Validation
## Maximum Output Request: Complete Experimental System

**Your Task:** Design and implement COMPLETE behavioral validation experiment that proves R_V contraction causes recursive-like text generation.

**Context:** We've proven geometric contraction (R_V < 1.0) at Layer 27 via activation patching (d=-3.56). Now we need to prove this geometry CAUSES behavior.

**Goal:** Close the geometry → behavior loop. When we patch recursive V-space into baseline prompts, generated text should become more recursive/introspective.

---

## 🎯 **WHAT I NEED FROM YOU**

### **1. Complete Experimental Design Document**
- Detailed protocol (step-by-step)
- Exact metric definitions (formulas, token lists)
- Statistical analysis plan (tests, thresholds, corrections)
- Expected results with confidence intervals
- Control conditions specification

### **2. Production-Ready Jupyter Notebook Script**
- Copy-paste ready for RunPod
- Works with HF models (Mistral-7B-Instruct-v0.2)
- Uses existing code from `mistral_L27_FULL_VALIDATION.py`
- Uses existing prompt pairs (n=151 from CSV)
- Generates text for 3 conditions: baseline, recursive, patched
- Computes all behavioral metrics
- Runs statistical analysis
- Outputs standardized CSV

### **3. Post-Experiment Analysis Script**
- Statistical tests (paired t-tests, effect sizes)
- Visualizations (distributions, comparisons)
- Correlation analysis (R_V delta vs behavioral markers)
- Results interpretation

### **4. Documentation**
- How to run the experiment
- What each metric means
- How to interpret results
- Troubleshooting guide

---

## 📋 **EXPERIMENTAL REQUIREMENTS**

### **Sample:**
- Use existing n=151 pairs from `mistral_L27_FULL_VALIDATION.py`
- Each pair: (baseline_prompt, recursive_prompt)
- For each pair, generate text under 3 conditions

### **Conditions:**
1. **Baseline (natural):** Generate with baseline prompt, no patching
2. **Recursive (natural):** Generate with recursive prompt, no patching  
3. **Patched (intervention):** Generate with baseline prompt + L27 V-space patch from recursive

### **Behavioral Metrics to Compute:**

**Metacognitive Marker Frequency:**
```python
metacognitive_tokens = ["I", "aware", "observe", "notice", "recognize", 
                        "self", "conscious", "thinking", "awareness", 
                        "introspect", "reflect", "perceive"]
frequency = count(metacognitive_tokens) / total_tokens
```

**Self-Reference Density:**
```python
self_mentions = ["I", "my", "myself", "me", "this response", 
                 "this answer", "I am", "I think", "I notice"]
density = count(self_mentions) / total_tokens
```

**Recursive Structure Score:**
```python
# Count nested self-mentions (e.g., "I notice that I am thinking")
nested_patterns = ["I notice that I", "I observe myself", 
                  "I am aware that I", "I recognize that I"]
score = count(nested_patterns) / count(self_mentions)
```

**Introspective Content Classification:**
```python
# Use LLM-based classification (or rule-based)
# Prompt: "Is this text introspective/self-referential? Yes/No"
introspective_score = classify_introspective(text)
```

### **Controls:**
- **Random patch control:** Patch with random noise → should NOT increase markers
- **Wrong-layer control:** Patch at L5 instead of L27 → should NOT increase markers
- **Baseline-only:** Natural baseline generation (reference)

### **Statistical Analysis:**
- Paired t-tests: patched vs baseline, patched vs recursive
- Effect sizes: Cohen's d for marker frequency differences
- Correlation: R_V delta vs behavioral marker increase
- Bonferroni correction for multiple comparisons

---

## 💻 **CODE REQUIREMENTS**

### **Must Work With:**
- Existing code: `mistral_L27_FULL_VALIDATION.py`
- Existing prompts: `n300_mistral_test_prompt_bank.py`
- Existing pairs: CSV from `mistral7b_n200_BULLETPROOF.csv`
- HF models on RunPod

### **Script Structure:**
```python
# behavioral_validation_experiment.py
# Copy-paste ready for RunPod Jupyter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from pathlib import Path
from mistral_L27_FULL_VALIDATION import (
    load_model, get_prompt_pairs, patch_v_at_layer_27
)

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda"
RESULTS_DIR = Path("results/behavioral_validation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load existing pairs
pairs_df = pd.read_csv("mistral7b_n200_BULLETPROOF.csv")
# Use n=151 valid pairs

def generate_text_with_patching(model, tokenizer, base_text, rec_text, 
                                 patch_layer=27, condition="patched"):
    """Generate text under different conditions"""
    # [Your implementation]
    pass

def compute_behavioral_metrics(text):
    """Compute all behavioral metrics"""
    # [Your implementation]
    pass

def run_statistical_analysis(results_df):
    """Run all statistical tests"""
    # [Your implementation]
    pass

def main():
    """Main experiment"""
    # [Your implementation]
    pass

if __name__ == "__main__":
    main()
```

### **Output CSV Format:**
```csv
pair_idx,base_id,rec_id,condition,generated_text,metacognitive_freq,
self_ref_density,recursive_score,introspective_score,rv_delta
```

---

## 🎯 **EXPECTED OUTPUTS**

### **1. Experimental Design Document** (`behavioral_validation_design.md`)
- Complete protocol
- Metric definitions
- Statistical plan
- Expected results

### **2. Main Script** (`behavioral_validation_experiment.py`)
- Production-ready code
- Copy-paste ready for Jupyter
- Full error handling
- Progress logging

### **3. Analysis Script** (`behavioral_analysis.py`)
- Statistical tests
- Visualizations
- Results interpretation

### **4. Documentation** (`BEHAVIORAL_VALIDATION_README.md`)
- How to run
- What metrics mean
- How to interpret
- Troubleshooting

---

## ✅ **SUCCESS CRITERIA**

- [ ] Script runs on RunPod with HF models
- [ ] Generates text for all 3 conditions
- [ ] Computes all 4 behavioral metrics
- [ ] Runs statistical analysis
- [ ] Outputs standardized CSV
- [ ] Includes error handling
- [ ] Includes progress logging
- [ ] Documentation complete

---

## 🚀 **MAXIMUM OUTPUT REQUEST**

**I want EVERYTHING:**
- Complete experimental design
- Production-ready code
- Analysis scripts
- Documentation
- Test on 10 pairs before full run
- Error handling for edge cases
- Progress bars and logging
- Resume capability (can continue if interrupted)

**Make it so complete that I can copy-paste into Jupyter and run immediately.**

---

**Start with experimental design, then implement code, then create analysis scripts, then write documentation. Deliver ALL of it.**

