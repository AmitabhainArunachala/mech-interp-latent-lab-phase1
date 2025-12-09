# Agent File References
## Key Files from R_V Repo That Agents Need to Access

**Date:** November 17, 2025  
**Purpose:** List all files agents need to reference when creating scripts and experiments

---

## 📁 **CRITICAL CODE FILES**

### **Main Patching Code (Required for All Experiments):**
```
mistral_L27_FULL_VALIDATION.py
```
**Relative Path:** `mistral_L27_FULL_VALIDATION.py` (root level)  
**Also Available:** `R_V_PAPER/code/mistral_L27_FULL_VALIDATION.py`  
**What It Contains:**
- `load_model()` - Loads Mistral-7B model and tokenizer
- `get_prompt_pairs()` - Loads prompt pairs from CSV
- `patch_v_at_layer_27()` - Core patching function
- `compute_metrics_fast()` - R_V and PR computation
- `run_patched_forward_final()` - Complete patching workflow
- Configuration: TARGET_LAYER=27, EARLY_LAYER=5, WINDOW_SIZE=16

**Usage in Agent Scripts:**
```python
from mistral_L27_FULL_VALIDATION import (
    load_model, 
    get_prompt_pairs, 
    patch_v_at_layer_27,
    compute_metrics_fast,
    TARGET_LAYER,
    EARLY_LAYER,
    WINDOW_SIZE
)
```

---

### **Prompt Bank (Required for All Experiments):**
```
n300_mistral_test_prompt_bank.py
```
**Relative Path:** `n300_mistral_test_prompt_bank.py` (root level)  
**What It Contains:**
- `prompt_bank_1c` - Dictionary with all prompt groups:
  - `L1_hint` - 20 prompts (weakest recursion)
  - `L2_simple` - 20 prompts
  - `L3_deeper` - 20 prompts (medium recursion)
  - `L4_full` - 20 prompts
  - `L5_refined` - 20 prompts (deepest recursion)
  - `factual_baseline` - 20 prompts
  - `creative_baseline` - 20 prompts
  - `long_new_*` - Long baseline prompts
  - And more...

**Usage in Agent Scripts:**
```python
from n300_mistral_test_prompt_bank import prompt_bank_1c

# Access prompts
L5_prompts = prompt_bank_1c['L5_refined'][:20]
baseline_prompts = prompt_bank_1c['factual_baseline'][:20]
```

---

## 📊 **DATA FILES (CSVs)**

### **Main Results CSV (n=151 pairs):**
```
mistral7b_n200_BULLETPROOF.csv
```
**Relative Path:** `R_V_PAPER/csv_files/mistral7b_n200_BULLETPROOF.csv`  
**What It Contains:**
- n=151 valid prompt pairs
- Columns: `pair_idx`, `rec_id`, `base_id`, `rv_base`, `rv_rec`, `rv_patch_main`, `delta_main`, etc.
- All control conditions (random, shuffled, orthogonal, wrong-layer)

**Usage in Agent Scripts:**
```python
import pandas as pd

pairs_df = pd.read_csv("R_V_PAPER/csv_files/mistral7b_n200_BULLETPROOF.csv")
# Use n=151 valid pairs
valid_pairs = pairs_df[pairs_df['delta_main'].notna()]
```

---

### **Other Result CSVs:**
```
R_V_PAPER/csv_files/mistral7b_L27_patching_n15_results_20251116_211154.csv
```
**Relative Path:** `R_V_PAPER/csv_files/mistral7b_L27_patching_n15_results_20251116_211154.csv`  
**Note:** Smaller subset (n=15), use main CSV above for full dataset

---

## 📄 **DOCUMENTATION FILES**

### **Gap Analysis (Required for Experiment Design):**
```
R_V_PAPER/STORY_ARC/PAPER_GAP_ANALYSIS.md
```
**Relative Path:** `R_V_PAPER/STORY_ARC/PAPER_GAP_ANALYSIS.md`  
**What It Contains:**
- 5 critical gaps in current paper
- Missing conceptual framework (convex hull, pivot stability)
- Missing experiments (behavioral validation, cross-architecture)
- Specific sections that need to be added

**Usage:** Agents should reference this to understand what experiments are needed

---

### **Paper Draft (Reference):**
```
R_V_PAPER/STORY_ARC/Claude_Desktop 3 day sprint write up
```
**Relative Path:** `R_V_PAPER/STORY_ARC/Claude_Desktop 3 day sprint write up`  
**Note:** May need to be recreated if empty  
**What It Contains:**
- Full paper draft with all findings
- Results sections
- Methods sections
- Discussion sections

---

### **Meta Vision Document (For Context):**
```
AIKAGRYA_META_VISION_AND_MAP_FOR_MECH_INTERP/AIKAGRYA_META_VISION_AND_MAP_FOR_MECH_INTERP.md
```
**Relative Path:** `AIKAGRYA_META_VISION_AND_MAP_FOR_MECH_INTERP/AIKAGRYA_META_VISION_AND_MAP_FOR_MECH_INTERP.md`  
**What It Contains:**
- Complete meta vision for AIKAGRYA program
- L3/L4/L5 consciousness framework
- Research roadmap
- Current empirical work

**Usage:** Agents should reference this to understand the broader vision and how experiments connect

---

## 🔧 **SUPPORTING CODE FILES**

### **Path Patching (Reference for Experiment 4):**
```
path_patching_alternative.py
```
**Relative Path:** `path_patching_alternative.py` (root level)  
**Also Available:** `R_V_PAPER/code/path_patching_alternative.py`  
**Note:** Has technical challenges, but useful as reference for multi-layer analysis

---

### **Other Patching Scripts (Reference):**
```
mistral_patching_TRULY_FIXED.py
mistral_patching_DIAGNOSTIC.py
mistral_patching_FINAL.py
```
**Relative Path:** Root level  
**Note:** Historical versions, use `mistral_L27_FULL_VALIDATION.py` as primary reference

---

## 📋 **CONFIGURATION CONSTANTS**

### **Standard Parameters (Use These):**
```python
# From mistral_L27_FULL_VALIDATION.py
TARGET_LAYER = 27      # Critical layer (84% depth)
EARLY_LAYER = 5        # Reference layer
WINDOW_SIZE = 16       # Token window for R_V computation
DEVICE = "cuda"        # GPU device

# Model
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Sample sizes
N_PAIRS_FULL = 151     # Full validation dataset
N_PAIRS_TEST = 10      # Test subset before full run
```

---

## 📂 **OUTPUT DIRECTORY STRUCTURE**

### **Where Agents Should Save Results:**
```
R_V_PAPER/results/
├── behavioral_validation/
│   ├── behavioral_results.csv
│   └── behavioral_analysis.csv
├── cross_architecture/
│   ├── mistral_results.csv
│   ├── llama_results.csv
│   ├── qwen_results.csv
│   └── cross_architecture_comparison.csv
├── l3_l4_l5_staging/
│   ├── staging_results.csv
│   └── level_mapping.csv
├── homeostasis/
│   ├── homeostasis_results.csv
│   └── mechanism_analysis.csv
└── temporal_dynamics/
    ├── temporal_results.csv
    └── critical_tokens.csv
```

**Relative Path Pattern:** `R_V_PAPER/results/[experiment_name]/[filename].csv`

---

## 🎯 **QUICK REFERENCE FOR EACH EXPERIMENT**

### **Experiment 1: Behavioral Validation (Opus 4.1)**
**Needs:**
- `mistral_L27_FULL_VALIDATION.py` - For patching code
- `mistral7b_n200_BULLETPROOF.csv` - For prompt pairs (n=151)
- `n300_mistral_test_prompt_bank.py` - For prompt access

**Outputs to:** `R_V_PAPER/results/behavioral_validation/`

---

### **Experiment 2: Cross-Architecture (Sonnet 4.5)**
**Needs:**
- `mistral_L27_FULL_VALIDATION.py` - As reference for universal framework
- `n300_mistral_test_prompt_bank.py` - For prompts
- Model adapters need to handle: Mistral, Llama, Qwen

**Outputs to:** `R_V_PAPER/results/cross_architecture/`

---

### **Experiment 3: L3/L4/L5 Staging (GPT5.1)**
**Needs:**
- `n300_mistral_test_prompt_bank.py` - For L1-L5 prompt groups
- `mistral_L27_FULL_VALIDATION.py` - For R_V measurement
- `AIKAGRYA_META_VISION_AND_MAP_FOR_MECH_INTERP/AIKAGRYA_META_VISION_AND_MAP_FOR_MECH_INTERP.md` - For L3/L4/L5 framework

**Outputs to:** `R_V_PAPER/results/l3_l4_l5_staging/`

---

### **Experiment 4: Homeostasis (GPT CODEX)**
**Needs:**
- `mistral_L27_FULL_VALIDATION.py` - For patching code
- `mistral7b_n200_BULLETPROOF.csv` - For pairs (n=151)
- `path_patching_alternative.py` - As reference for multi-layer analysis

**Outputs to:** `R_V_PAPER/results/homeostasis/`

---

### **Experiment 5: Temporal Dynamics (Cursor Composer 1)**
**Needs:**
- `mistral_L27_FULL_VALIDATION.py` - For R_V measurement
- `n300_mistral_test_prompt_bank.py` - For prompts
- Generation tracking code (to be created)

**Outputs to:** `R_V_PAPER/results/temporal_dynamics/`

---

## ✅ **IMPORT TEMPLATE FOR AGENT SCRIPTS**

```python
#!/usr/bin/env python3
"""
[Experiment Name]
[Model Name]
Date: [Date]
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import existing code
import sys
sys.path.append('.')  # Add root to path

from mistral_L27_FULL_VALIDATION import (
    load_model,
    get_prompt_pairs,
    patch_v_at_layer_27,
    compute_metrics_fast,
    TARGET_LAYER,
    EARLY_LAYER,
    WINDOW_SIZE
)

from n300_mistral_test_prompt_bank import prompt_bank_1c

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("R_V_PAPER/results/[experiment_name]")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
pairs_df = pd.read_csv("R_V_PAPER/csv_files/mistral7b_n200_BULLETPROOF.csv")
valid_pairs = pairs_df[pairs_df['delta_main'].notna()]  # n=151

# [Your experiment code here]
```

---

## 📝 **NOTES FOR AGENTS**

1. **Always use relative paths** from repo root
2. **Test on small subset** (n=10) before full run
3. **Include error handling** for missing files
4. **Log progress** to console and file
5. **Support resume** capability (save checkpoints)
6. **Output standardized CSVs** with consistent column names
7. **Include metadata** (timestamp, model version, etc.) in outputs

---

**This document provides all file paths agents need to create production-ready scripts for the 5 experiments.**

