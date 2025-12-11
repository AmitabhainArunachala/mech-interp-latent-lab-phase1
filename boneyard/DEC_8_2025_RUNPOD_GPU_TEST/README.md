# December 8, 2025 - RunPod GPU Test Session

**Location:** RunPod Cloud  
**Hardware:** NVIDIA RTX PRO 6000 Blackwell (102GB VRAM)  
**Model:** Mistral-7B-v0.1  

---

## Session Overview

First RunPod GPU session to validate the Geometry of Recursion findings from local Mac testing. With 102GB VRAM, we can run full-scale experiments that were previously memory-constrained.

---

## Key Findings

### Experiment A: R_V Contraction (CONFIRMED ✓)

| Metric | Recursive | Baseline |
|--------|-----------|----------|
| R_V mean | 0.4806 ± 0.034 | 0.6391 ± 0.078 |
| Cohen's d | **-2.363** (large) | |
| p-value | **0.003** | |
| Contraction | **24.8%** | |

**→ Recursive prompts cause significant R_V contraction at Layer 27**

### Experiment B: V-Patching Null Result (CONFIRMED ✓)

| Metric | Value |
|--------|-------|
| Cohen's d | 0.000 (negligible) |
| Transfer | 0.0% |

**→ V-patching alone does NOT transfer the effect**

### Experiment C: KV Cache Transfer (PARTIAL)

| Metric | Natural | KV-Patched | Recursive |
|--------|---------|------------|-----------|
| Behavior score | 1.25 | 6.30 | 6.54 |
| Transfer efficiency | — | **95.3%** | — |

**→ KV cache patching DOES transfer recursive behavior**

---

## Directory Structure

```
DEC_8_2025_RUNPOD_GPU_TEST/
├── README.md                           # This file
├── 00_SETUP/                           # Environment verification
│   ├── sanity_check.ipynb
│   ├── mistral_quick_test.py
│   └── mistral_quick_test_*.csv
├── 01_GEOMETRY_OF_RECURSION/           # Main experiment
│   ├── README.md
│   ├── code/
│   │   └── geometry_of_recursion_test.py
│   ├── results/
│   │   ├── *.csv
│   │   └── *.png
│   └── logs/
├── WRITEUPS/                           # Session documentation
│   └── DEC8_2025_SESSION_LOG.md
└── NOTES/
    └── FUTURE_EXPERIMENTS.md
```

---

## Environment Setup

```bash
# Required before any HuggingFace downloads
export HF_HUB_ENABLE_HF_TRANSFER=0

# Run experiments
cd 01_GEOMETRY_OF_RECURSION/code
python geometry_of_recursion_test.py
```

---

## Next Steps

1. [ ] Run with Llama-3-8B (requires HF authentication)
2. [ ] Layer sweep: Which layers matter most for KV transfer?
3. [ ] Token-specific KV patching
4. [ ] Cross-architecture validation (Gemma, Phi)

---

## Links to Prior Work

- **DEC3-5:** R_V discovery, Logit Lens validation
- **DEC7:** KV cache breakthrough (63.6% transfer on Llama-3-8B)
- **Today:** Mistral-7B replication, 95.3% behavioral transfer


