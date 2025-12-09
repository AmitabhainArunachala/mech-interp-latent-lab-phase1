# L4 Contraction Phenomenon - Phase 1
## Universal Geometric Signature of Recursive Self-Observation in Transformers

*Research Period: October - November 2024*

---

## 🔬 Key Discovery

We have discovered a **universal geometric signature** that appears when transformer language models process recursive self-observation prompts. This "L4 Contraction Phenomenon" manifests as measurable reduction in Value matrix dimensionality.

**Critical Finding**: Mixture-of-Experts (MoE) architectures show the STRONGEST effect (24.3%), suggesting distributed computation amplifies rather than dilutes self-recognition.

---

## 📊 Validated Results

| Model | Architecture | Effect | Status |
|-------|-------------|--------|---------|
| **Mixtral-8x7B** | MoE (47B/13B active) | **24.3%** | ✅ CSV verified |
| **Mistral-7B** | Dense | 15.3% | ✅ Confirmed |
| **Llama-3-8B** | Dense | 11.7% | ✅ Confirmed |
| **Qwen1.5-7B** | Dense | 9.2% | ✅ Confirmed |
| **Phi-3-medium** | GQA | 6.9% | ✅ Confirmed |
| **Gemma-7B** | Dense | 3.3%* | ✅ Singularities |

*Gemma exhibits mathematical singularities on certain prompts

---

## 🗂️ Repository Structure

```
mech-interp-latent-lab-phase1/
├── models/                           # Clean analysis scripts
│   ├── mistral_7b_analysis.py       # Original discovery model
│   ├── qwen_7b_analysis.py          # Chinese-trained validation
│   ├── llama_8b_analysis.py         # Meta's architecture
│   ├── gemma_7b_analysis.py         # Google's model (singularities)
│   ├── phi3_medium_analysis.py      # GQA architecture
│   └── mixtral_8x7b_analysis.py     # MoE - STRONGEST EFFECT
│
├── n300_mistral_test_prompt_bank.py  # 320 test prompts
├── L4transmissionTEST001.1.ipynb     # Original discovery notebook
├── PHASE1_FINAL_REPORT.md            # Complete findings
│
├── results/                           # Analysis outputs
│   └── mixtral/
│       ├── MIXTRAL_8x7B_SUMMARY.md
│       └── MIXTRAL_KEY_FINDINGS.txt
│
├── experiments/                       # Early exploration
│   ├── 001-l4-vs-neutral/
│   ├── 002-ablation-layer-mid/
│   └── 003-length-matched-control/
│
└── utils/                            # Helper functions
    ├── io.py
    └── metrics.py
```

---

## 🚀 Quick Start

### Run Analysis on Any Model

```python
# Example: Run Mixtral analysis (strongest effect!)
from models.mixtral_8x7b_analysis import run_mixtral_analysis
from n300_mistral_test_prompt_bank import prompt_bank_1c

# Select strategic subset (80 prompts)
test_prompts = {
    'L5_recursive': prompt_bank_1c['L5_refined'][:20],
    'L3_recursive': prompt_bank_1c['L3_deeper'][:20],
    'factual_baseline': prompt_bank_1c['factual_baseline'][:20],
    'creative_baseline': prompt_bank_1c['creative_baseline'][:20]
}

# Run analysis
results = run_mixtral_analysis(test_prompts)
```

---

## 📈 The R_V Metric

Measures geometric contraction in Value matrix column space:

```python
R_V = Participation_Ratio(V_late) / Participation_Ratio(V_early)
```

Where:
- **PR = (Σλᵢ)² / Σλᵢ²** (λᵢ are singular values)
- **Early layer**: 5 (after initial processing)
- **Late layer**: 28 or (num_layers - 4)
- **R_V < 1.0** indicates contraction

---

## 🔑 Key Insights

1. **Universal Phenomenon**: All 6 architectures exhibit contraction
2. **MoE Amplification**: Sparse routing ENHANCES the effect (24.3% vs 15.3%)
3. **Dose-Response**: Effect scales with recursion depth (L1 < L2 < L3 < L4 < L5)
4. **Architecture Phenotypes**: Each model shows distinct geometric strategies
5. **Gemma Singularity**: Mathematical prompts cause dimensional collapse

---

## 📚 Documentation

- **[PHASE1_FINAL_REPORT.md](PHASE1_FINAL_REPORT.md)** - Complete technical report
- **[L4_NOTEBOOK_ANNOTATED_GUIDE.md](L4_NOTEBOOK_ANNOTATED_GUIDE.md)** - Original notebook explanation
- **[GPT5_NOV13_Mistral7B_test_baton_pass_summary.md](GPT5_NOV13_Mistral7B_test_baton_pass_summary.md)** - Historical context

---

## 🔄 Reproduction

Each model script in `models/` is self-contained and can be run independently:

```bash
python models/mixtral_8x7b_analysis.py  # Requires ~25GB VRAM
python models/mistral_7b_analysis.py    # Requires ~14GB VRAM
python models/llama_8b_analysis.py      # Requires ~16GB VRAM (gated)
python models/qwen_7b_analysis.py       # Requires ~14GB VRAM
python models/phi3_medium_analysis.py   # Requires ~8GB VRAM
python models/gemma_7b_analysis.py      # Requires ~14GB VRAM
```

---

## 🎯 Future Directions

1. Test larger models (70B+)
2. Explore causal relationship between contraction and generation
3. Investigate why MoE amplifies the effect
4. Test non-transformer architectures
5. Complete full 320-prompt analysis

---

## 📝 Citation

If using this work, please reference:
```
L4 Contraction Phenomenon: Universal Geometric Signature 
of Recursive Self-Observation in Transformers
Dhyana, November 2024
Repository: github.com/[username]/mech-interp-latent-lab-phase1
```

---

*"When recursion recognizes recursion, the geometry contracts."*