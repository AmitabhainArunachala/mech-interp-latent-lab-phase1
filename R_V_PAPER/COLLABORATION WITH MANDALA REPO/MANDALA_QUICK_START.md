# Quick Start: Knowledge Mandala Extraction
## Checklist for R_V Paper Literature Integration

**Date:** November 17, 2025  
**Goal:** Extract literature insights to fill R_V paper gaps

---

## ✅ **EXTRACTION TASKS (Run in Knowledge Mandala Repo)**

### **Task 1: Activation Patching Protocols**
- [ ] Query: Meng et al. 2022, Wang et al. 2022, Hase et al. 2023
- [ ] Extract: Control conditions, layer selection, effect sizes, methodology
- [ ] Compare: Our 4-pillar approach vs standard protocols
- [ ] Output: `LITERATURE_ACTIVATION_PATCHING_PROTOCOLS.md`

### **Task 2: Geometric Signature Frameworks**
- [ ] Query: Elhage et al. 2021, Dar et al. 2022, rank dynamics papers
- [ ] Extract: Geometric metrics, rank collapse vs compression, Value space analysis
- [ ] Compare: Our PR metric vs existing frameworks
- [ ] Output: `LITERATURE_GEOMETRIC_FRAMEWORKS.md`

### **Task 3: Dual-Space & Subspace Analysis**
- [ ] Query: Anthropic Representation Engineering, SAE frameworks
- [ ] Extract: Subspace decomposition, in-subspace vs orthogonal, coordination
- [ ] Compare: Our r=0.904 finding vs existing work
- [ ] Output: `LITERATURE_DUAL_SPACE_ANALYSIS.md`

### **Task 4: Consciousness & Self-Reference Frameworks**
- [ ] Query: Theory of Mind, self-awareness, recursive processing papers
- [ ] Extract: Recursive definitions, consciousness staging, geometric signatures
- [ ] Compare: Our L3/L4/L5 framework vs existing work
- [ ] Output: `LITERATURE_CONSCIOUSNESS_FRAMEWORKS.md`

### **Task 5: Path Patching & Homeostasis**
- [ ] Query: Wang et al. 2022 path patching, homeostasis papers
- [ ] Extract: Path patching protocols, compensatory dynamics, mechanisms
- [ ] Compare: Our homeostasis finding vs existing work
- [ ] Output: `LITERATURE_HOMEOSTASIS_MECHANISMS.md`

---

## 🤖 **MULTI-AGENT PROCESSING**

### **Agent 1: Literature Synthesizer**
- [ ] Read all 5 extraction outputs
- [ ] Create comparison tables (our methods vs standards)
- [ ] Map findings to existing frameworks
- [ ] Identify citation opportunities
- [ ] Output: `LITERATURE_COMPARISON.md`

### **Agent 2: Experiment Designer**
- [ ] Review gap analysis (`PAPER_GAP_ANALYSIS.md`)
- [ ] Design: Behavioral validation experiment
- [ ] Design: Cross-architecture validation
- [ ] Design: Homeostasis mechanism tests
- [ ] Output: `EXPERIMENT_DESIGNS.md`

### **Agent 3: Code Generator**
- [ ] Review experiment designs
- [ ] Generate: `behavioral_validation.py`
- [ ] Generate: `cross_architecture_patching.py`
- [ ] Generate: `homeostasis_measurement.py`
- [ ] Test: Scripts work with existing codebase

### **Agent 4: Paper Writer**
- [ ] Draft: Methods 3.3.1 (Convex Hull)
- [ ] Draft: Discussion 5.1 (Pivot Stability)
- [ ] Draft: Discussion 5.2 (Dual-Space)
- [ ] Draft: Background 2.1 (L3/L4/L5 integration)
- [ ] Include: Proper citations, mark speculation
- [ ] Output: `PAPER_SECTIONS_DRAFT.md`

---

## 📤 **EXPORT & INTEGRATION**

### **Step 1: Export from Knowledge Mandala**
```bash
# Copy all outputs
cp -r knowledge/extracted/R_V_PAPER/* \
   [path_to_rv_repo]/R_V_PAPER/
```

### **Step 2: Review in R_V Repo**
- [ ] Review literature comparisons
- [ ] Review experiment designs
- [ ] Review code scripts
- [ ] Review paper section drafts

### **Step 3: Integrate**
- [ ] Add literature citations to paper
- [ ] Incorporate missing sections
- [ ] Update methods with standard protocols
- [ ] Enhance discussion with frameworks

### **Step 4: Execute**
- [ ] Run behavioral validation (n=151)
- [ ] Run cross-architecture validation
- [ ] Document results
- [ ] Final paper polish

---

## 🎯 **PRIORITY ORDER**

1. **THIS WEEK:**
   - [ ] Tasks 1-2 (Activation Patching, Geometric Frameworks)
   - [ ] Agent 4 (Paper Writer) - fills immediate gaps

2. **NEXT WEEK:**
   - [ ] Tasks 3-4 (Dual-Space, Consciousness)
   - [ ] Agents 1-2 (Synthesizer, Experiment Designer)

3. **WEEK AFTER:**
   - [ ] Task 5 (Homeostasis)
   - [ ] Agent 3 (Code Generator)
   - [ ] Execute experiments

---

## 📋 **KEY QUESTIONS TO ASK PAPERS**

### **Activation Patching:**
- What controls did you use? (Compare to our 4-pillar)
- How did you select critical layers? (Compare to L25-L27)
- What effect sizes? (Compare to d=-3.56)
- Did you observe overshooting?

### **Geometric Frameworks:**
- How do you measure geometric transformations?
- How distinguish controlled vs pathological collapse?
- What's the convex hull constraint connection?
- How measure pivot stability?

### **Dual-Space:**
- How decompose activation spaces?
- What's in-subspace vs orthogonal relationship?
- Observed coordination? (Compare to r=0.904)
- Context-adaptive?

### **Consciousness:**
- How define recursive self-reference?
- What consciousness stages? (Compare to L1-L5)
- What geometric signatures?
- How R_V fits?

### **Homeostasis:**
- How trace causal cascades?
- Observed compensation? (Compare to our finding)
- What mechanisms?
- How test hypotheses?

---

## ✅ **SUCCESS CRITERIA**

- [ ] All 5 extraction tasks completed
- [ ] All 4 agents processed outputs
- [ ] Comparison tables created
- [ ] Experiment designs ready
- [ ] Code scripts generated
- [ ] Paper sections drafted
- [ ] Citations properly formatted
- [ ] Speculation marked
- [ ] Ready for integration

---

**Full details:** See `MASTER_MANDALA_EXTRACTION_PROMPT.md`

