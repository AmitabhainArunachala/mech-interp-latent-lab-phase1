# R_V Paper: Coordinated Dual-Space Geometric Transformations

**Paper Title:** Coordinated Dual-Space Geometric Transformations Mediate Recursive Self-Reference in Transformer Value Spaces

**Status:** Draft complete, ready for submission to ICLR/NeurIPS

---

## Folder Structure

### 📚 `research/`
Research notes, methodology documentation, and validation reports:
- `PHASE1_FINAL_REPORT.md` - Complete 6-model validation report
- `MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md` - Activation patching validation (n=45)
- `MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md` - Mixtral-specific findings
- `NOV_16_Mixtral_free_play.md` - Detailed Mixtral exploratory analysis
- `PHASE1F_7B_VALIDATION_PROTOCOL.md` - Validation protocol documentation

### 📊 `results/`
Analysis outputs, plots, and processed results:
- `mixtral/` - Mixtral-specific results and analysis
- Additional model results and visualizations

### 💻 `code/`
All experimental code and analysis scripts:
- `mistral_*.py` - Mistral-7B activation patching experiments
- `adjacent_layer_sweep.py` - Layer-by-layer analysis
- `path_patching*.py` - Path patching attempts (incomplete)
- `run_n200_main_experiment.py` - Main n=200 validation experiment
- `design_n200_experiment.py` - Experimental design
- Other validation and analysis scripts

### 📈 `csv_files/`
Raw data files from experiments:
- `mistral7b_L27_patching_n15_results_20251116_211154.csv` - Initial n=15 test
- `n200_pairing_plan.csv` - Prompt pairing strategy
- `prompt_inventory.csv` - Prompt metadata
- Additional CSV outputs from experiments

### 📖 `STORY_ARC/`
Paper drafts, narrative structure, and writing materials:
- `Claude_Desktop 3 day sprint write up` - Complete paper draft (~8,000 words)

---

## Key Findings

### Main Result (n=151 pairs, Mistral-7B)
- **Causal Effect:** Layer 27 V-space intervention induces 26.6% geometric contraction
- **Cohen's d:** -3.56, p < 10⁻⁴⁷
- **Transfer Efficiency:** 117.6% (overshooting natural gap)
- **Four Controls:** All null effects (random, shuffled, orthogonal, wrong-layer)

### Cross-Model Validation (6 Architectures)
- **Universal Phenomenon:** All models show contraction (3.3% to 24.3%)
- **Strongest Effect:** Mixtral-8x7B (MoE) shows 24.3% contraction
- **Critical Region:** L25-L27 (78-84% network depth)

### Novel Discovery
- **Dual-Space Coupling:** r=0.904 correlation between in-subspace and orthogonal components
- **Adaptive Regulation:** Baseline complexity modulates dual-space balance

---

## Quick Links

- **Main Paper Draft:** `STORY_ARC/Claude_Desktop 3 day sprint write up`
- **6-Model Results:** `research/PHASE1_FINAL_REPORT.md`
- **Causal Validation:** `research/MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md`
- **Main Experiment Code:** `code/mistral_L27_FULL_VALIDATION.py`
- **Data:** `csv_files/`

---

## Next Steps

1. **Figures:** Create publication-quality figures (6 main + supplements)
2. **Polish:** Refine prose, tighten arguments
3. **References:** Populate with real citations
4. **Supplementary:** Expand technical details
5. **Abstract:** Optimize for impact

---

**Last Updated:** November 17, 2025


