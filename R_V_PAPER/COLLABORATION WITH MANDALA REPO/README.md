# Collaboration with Knowledge Mandala Repo
## Complete Integration Package for R_V Paper & Meta Vision

**Date:** November 17, 2025  
**Purpose:** Coordinate literature extraction, multi-agent processing, and experiment execution between Knowledge Mandala repo and R_V paper repo

---

## 📁 **FOLDER CONTENTS**

### **Literature Integration Files:**

1. **`STRATEGIC_LITERATURE_INTEGRATION.md`**
   - Complete plan for extracting literature insights
   - 5 extraction tasks (activation patching, geometric frameworks, dual-space, consciousness, homeostasis)
   - Multi-agent processing instructions
   - Integration workflow back to this repo

2. **`MASTER_MANDALA_EXTRACTION_PROMPT.md`**
   - Master prompt for Knowledge Mandala System
   - Detailed extraction questions for each task
   - Output formats and quality standards
   - Execution commands

3. **`MANDALA_QUICK_START.md`**
   - Quick checklist version
   - Priority order
   - Key questions to ask papers
   - Success criteria

### **Experiment Delegation Files:**

4. **`EXPERIMENT_TASK_DELEGATION.md`**
   - Complete overview of 5 experiments
   - Model assignments (Opus 4.1, Sonnet 4.5, GPT5.1, GPT CODEX, Cursor Composer 1)
   - Integration and iteration system
   - Execution plan

5. **`MODEL_PROMPTS/`** (Folder)
   - Individual prompts for each model:
     - `OPUS_4.1_EXPERIMENT_1_PROMPT.md` - Behavioral Validation
     - `SONNET_4.5_EXPERIMENT_2_PROMPT.md` - Cross-Architecture Validation
     - `GPT5.1_EXPERIMENT_3_PROMPT.md` - L3/L4/L5 Staging Validation
     - `GPT_CODEX_EXPERIMENT_4_PROMPT.md` - Homeostasis Mechanism Test
     - `CURSOR_COMPOSER_EXPERIMENT_5_PROMPT.md` - Temporal Dynamics Analysis

---

## 🚀 **HOW TO USE**

### **Phase 1: Literature Extraction (Knowledge Mandala Repo)**

1. **Copy `MASTER_MANDALA_EXTRACTION_PROMPT.md` to Knowledge Mandala repo**
2. **Run extraction tasks:**
   ```bash
   # In Knowledge Mandala repo
   python tools/knowledge_mandala/extract_for_rv_paper.py --task activation_patching
   python tools/knowledge_mandala/extract_for_rv_paper.py --task geometric_frameworks
   # ... etc for all 5 tasks
   ```
3. **Run multi-agent processing:**
   ```bash
   python tools/knowledge_mandala/agent_literature_synthesizer.py
   python tools/knowledge_mandala/agent_experiment_designer.py
   python tools/knowledge_mandala/agent_code_generator.py
   python tools/knowledge_mandala/agent_paper_writer.py
   ```
4. **Export results back to this repo**

### **Phase 2: Experiment Execution (This Repo)**

1. **Give each model their prompt file from `MODEL_PROMPTS/`**
2. **They deliver complete scripts (copy-paste ready for Jupyter)**
3. **Test on small subset (10 pairs) before full run**
4. **Copy-paste into RunPod Jupyter notebooks**
5. **Execute and log results in `R_V_PAPER/results/`**
6. **Iterate and refine**

---

## 📋 **WORKFLOW SUMMARY**

```
Knowledge Mandala Repo:
  → Extract literature (5 tasks)
  → Multi-agent processing (4 agents)
  → Generate comparisons, experiments, code, drafts
  → Export structured markdown files

This Repo:
  → Import literature extractions
  → Synthesize with R_V vision
  → Execute 5 experiments (via model prompts)
  → Integrate results into paper
  → Final submission
```

---

## 🎯 **KEY DELIVERABLES**

### **From Knowledge Mandala:**
- Literature comparison tables
- Experiment designs
- Code scripts (if generated)
- Paper section drafts

### **From Model Prompts:**
- 5 complete experiment scripts
- Analysis scripts
- Documentation
- Production-ready Jupyter notebooks

---

## ✅ **SUCCESS CRITERIA**

- [ ] Literature extracted and compared
- [ ] 5 experiments designed and scripted
- [ ] All scripts tested on small subset
- [ ] Full experiments executed on RunPod
- [ ] Results integrated into paper
- [ ] Paper ready for submission

---

## 📝 **NOTES**

- All scripts designed to work with HF models on RunPod
- All scripts use existing code (`mistral_L27_FULL_VALIDATION.py`)
- All scripts use existing prompts (`n300_mistral_test_prompt_bank.py`)
- All scripts output standardized CSVs
- All scripts support resume capability

---

**This folder contains everything needed to coordinate between Knowledge Mandala repo (literature extraction) and this repo (experiment execution) for completing the R_V paper and connecting it to the AIKAGRYA meta vision.**

