# Quick Navigation: Key Files and Findings

*Last Updated: November 19, 2025*

---

## 🎯 Core Discoveries

### The Universal Contraction
- **Finding:** Recursive self-reference causes measurable geometric contraction
- **Evidence:** 6+ architectures, p < 10⁻⁶, Cohen's d = -4.5
- **Report:** [`R_V_PAPER/research/PHASE1_FINAL_REPORT.md`](../R_V_PAPER/research/PHASE1_FINAL_REPORT.md)

### Pythia Confirmation
- **Finding:** 29.8% contraction (stronger than Mistral's 15%)
- **Evidence:** 320 prompts, 100% valid, t = -13.89
- **Report:** [`R_V_PAPER/research/PHASE_1C_PYTHIA_RESULTS.md`](../R_V_PAPER/research/PHASE_1C_PYTHIA_RESULTS.md)

### MoE Amplification
- **Finding:** Mixtral shows 24.3% contraction (strongest effect)
- **Mechanism:** Expert 5 creates high-dim features → downstream compression
- **Report:** [`R_V_PAPER/research/MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md`](../R_V_PAPER/research/MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md)

---

## 📁 Key Directories

### Research Documentation
- [`R_V_PAPER/research/`](../R_V_PAPER/research/) - All Phase 1 reports
- [`R_V_PAPER/code/`](../R_V_PAPER/code/) - Validated measurement code
- [`R_V_PAPER/csv_files/`](../R_V_PAPER/csv_files/) - Data artifacts

### Project Structures
- [`SUBSYSTEM_EMERGENCE_PYTHIA/`](../SUBSYSTEM_EMERGENCE_PYTHIA/) - Next phase vision
- [`SUBSYSTEM_2D_MAP_COMPLETION/`](../SUBSYSTEM_2D_MAP_COMPLETION/) - Sprint workspace

### Core Assets
- [`n300_mistral_test_prompt_bank.py`](../n300_mistral_test_prompt_bank.py) - 320 prompts
- [`models/mistral_7b_analysis.py`](../models/mistral_7b_analysis.py) - Original measurement code

---

## 🔬 Key Experiments

### Activation Patching (Causal)
- **File:** [`R_V_PAPER/csv_files/mistral7b_L27_patching_n15_results_20251116_211154.csv`](../R_V_PAPER/csv_files/mistral7b_L27_patching_n15_results_20251116_211154.csv)
- **Finding:** Patching transfers geometric signature (R_V drops 1.08 → 0.89)
- **Status:** Causal link established

### Layer 27 Snap Analysis
- **File:** [`R_V_PAPER/research/MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md`](../R_V_PAPER/research/MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md)
- **Finding:** 18/20 L5 prompts snap at Layer 27
- **Interpretation:** Late "decision corridor" where contraction occurs

### Expert Routing Probe
- **Finding:** Expert 5 over-weighted in recursive prompts (19.4% vs 12.5%)
- **Paradox:** Higher routing entropy, yet stronger contraction
- **Resolution:** Two-stage process (feature creation → consolidation)

---

## 📊 Key Metrics

| Metric | Value | Source |
|--------|-------|--------|
| **L5 R_V (Pythia)** | 0.564 ± 0.045 | Phase 1C |
| **Baseline R_V (Pythia)** | 0.804 ± 0.053 | Phase 1C |
| **Contraction (Pythia)** | 29.8% | Phase 1C |
| **Contraction (Mistral)** | 15.3% | Phase 1 |
| **Contraction (Mixtral)** | 24.3% | Phase 1 |
| **Cohen's d (Pythia)** | -4.51 | Phase 1C |
| **p-value (Pythia)** | < 10⁻⁶ | Phase 1C |

---

## 🚀 Next Steps

### Immediate
1. **Size Hypothesis:** Test Pythia-{160M → 12B}
2. **Developmental Sweep:** Checkpoints 0 → 143k
3. **Subsystem Mapping:** Creative, planning, uncertainty

### Short-term
4. **Behavioral Validation:** Does geometry → text generation?
5. **Cross-Architecture:** GPT-2, Llama-2, BERT
6. **Mechanistic Analysis:** Which heads/layers drive it?

---

## 🔗 Quick Links

- [Full Living Map](./LIVING_MAP.md) - Comprehensive exploration
- [Phase 1 Report](../R_V_PAPER/research/PHASE1_FINAL_REPORT.md) - Complete findings
- [Phase 1C Results](../R_V_PAPER/research/PHASE_1C_PYTHIA_RESULTS.md) - Pythia validation
- [Mixtral Analysis](../R_V_PAPER/research/MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md) - Deep mechanism

---

🌀

