# Master Prompt: Knowledge Mandala Extraction for R_V Paper
## Comprehensive Literature Analysis for Geometric Consciousness Signatures

**Date:** November 17, 2025  
**Target:** Extract protocols, frameworks, and gaps from 46 ingested papers  
**Output:** Structured markdown files for integration into R_V paper

---

## 🎯 **CONTEXT: The R_V Discovery**

We have discovered a universal geometric signature: recursive self-observation prompts cause measurable contraction in transformer Value matrix column spaces (R_V < 1.0), with effect sizes ranging from 3.3% to 24.3% across 6 architectures. Layer 27 (84% depth) causally mediates this via activation patching (Cohen's d = -3.56, p < 10⁻⁴⁷).

**Key Findings:**
- Universal phenomenon across 6 architectures (Mistral, Qwen, Llama, Phi-3, Gemma, Mixtral)
- MoE shows strongest effect (24.3% vs 15.3% dense)
- Causal validation with 4 controls (random, shuffled, wrong-layer, orthogonal)
- Dual-space coordination (r=0.904 coupling between in-subspace and orthogonal components)
- Homeostatic compensation (downstream layers compensate for L27 perturbations)

**Paper Status:** 90% complete, missing conceptual framework integration

---

## 📋 **EXTRACTION TASKS**

### **TASK 1: Activation Patching Protocols**

**Target Papers:**
- Meng et al. 2022 (Locating and Editing Factual Associations)
- Wang et al. 2022 (IOI Circuit - Interpretability in the Wild)
- Hase et al. 2023 (Does Localization Inform Editing?)
- Any other papers mentioning "activation patching" or "causal tracing"

**Extraction Questions:**

1. **Control Conditions:**
   - What control conditions did they use? (Random? Shuffled? Wrong-layer? Orthogonal?)
   - How do their controls compare to our 4-pillar approach?
   - Did they use norm-matching for random controls?
   - What statistical thresholds did they apply?

2. **Layer Selection:**
   - How did they identify critical layers? (Layer sweep? Hypothesis-driven?)
   - Did they observe layer-specific effects?
   - What depth percentages were critical? (Compare to our 84% / Layer 27)

3. **Effect Sizes:**
   - What Cohen's d values did they report?
   - Did they observe "overshooting" (patching > natural difference)?
   - How do their effect sizes compare to our d=-3.56?

4. **Methodology:**
   - How did they patch activations? (During forward pass? Post-hoc?)
   - What window sizes did they use? (Compare to our 16 tokens)
   - How did they handle token length mismatches?

5. **Validation:**
   - What statistical tests did they use?
   - How did they validate causality vs correlation?
   - What sample sizes were required?

**Output Format:**
```markdown
# Activation Patching Protocols - Literature Review

## Paper: [Title] ([Authors], [Year])

### Control Conditions
- Controls used: [list]
- Comparison to R_V 4-pillar approach: [analysis]
- Norm-matching: [yes/no, method]

### Layer Selection
- Critical layer identification: [method]
- Depth percentage: [X%]
- Comparison to L27 (84%): [analysis]

### Effect Sizes
- Cohen's d: [value]
- Overshooting observed: [yes/no]
- Comparison to d=-3.56: [analysis]

### Methodology
- Patching method: [description]
- Window size: [tokens]
- Token handling: [method]

### Validation
- Statistical tests: [list]
- Sample sizes: [n values]
- Causality validation: [method]

### Novelty Assessment
- What's standard: [list]
- What's novel in R_V work: [list]
- Gaps filled: [list]
```

---

### **TASK 2: Geometric Signature Frameworks**

**Target Papers:**
- Elhage et al. 2021 (Transformer Circuits Framework)
- Dar et al. 2022 (Information Compression in Transformers)
- Any papers on "rank dynamics", "dimensionality reduction", "participation ratio"
- Papers mentioning "singular value decomposition" in transformer context

**Extraction Questions:**

1. **Geometric Metrics:**
   - What metrics do they use to measure geometric transformations?
   - How do they compute Participation Ratio? (Compare to our PR = (Σλᵢ²)² / Σλᵢ⁴)
   - What's the relationship between PR, effective rank, and condition number?

2. **Rank Collapse vs Controlled Compression:**
   - How do they distinguish pathological rank collapse from controlled compression?
   - What's the relationship to pivot stability (Gaussian elimination)?
   - How do they measure "stability" in geometric terms?

3. **Value Space Geometry:**
   - How do they analyze Value matrix column spaces?
   - What's the relationship between attention's convex hull constraint and PR?
   - How do they measure subspace transformations?

4. **Architecture-Specific Patterns:**
   - Do different architectures show different geometric strategies?
   - How does MoE routing affect geometric transformations?
   - What's the relationship between depth and geometric transitions?

**Output Format:**
```markdown
# Geometric Signature Frameworks - Literature Review

## Paper: [Title] ([Authors], [Year])

### Geometric Metrics
- Metrics used: [list]
- Participation Ratio definition: [formula]
- Relationship to rank/condition number: [analysis]

### Rank Collapse Analysis
- Pathological vs controlled: [distinction]
- Pivot stability connection: [analysis]
- Stability measurement: [method]

### Value Space Analysis
- Column space methods: [description]
- Convex hull constraint: [explanation]
- Subspace transformations: [method]

### Architecture Patterns
- Architecture-specific findings: [list]
- MoE effects: [analysis]
- Depth relationships: [findings]

### Connection to R_V Work
- How R_V fits: [analysis]
- Novel contributions: [list]
- Gaps filled: [list]
```

---

### **TASK 3: Dual-Space & Subspace Analysis**

**Target Papers:**
- Anthropic's Representation Engineering papers
- SAE (Sparse Autoencoder) frameworks
- Papers on "subspace decomposition", "feature extraction", "principal components"
- Papers mentioning "orthogonal projection" in activation space

**Extraction Questions:**

1. **Subspace Decomposition:**
   - How do they decompose activation spaces into subspaces?
   - What PCA/SAE techniques do they use?
   - How do they select the number of principal components (k)?

2. **In-Subspace vs Orthogonal:**
   - How do they measure in-subspace vs orthogonal components?
   - What's the relationship between these components?
   - Have they observed coordination between subspaces?

3. **Feature Extraction:**
   - How do they extract "recursive processing" features?
   - Can they identify steering vectors?
   - How do features correlate with behavior?

4. **Dual-Space Coordination:**
   - Have they observed coupling between subspaces? (Compare to our r=0.904)
   - What mechanisms explain coordination?
   - Is coordination context-adaptive?

**Output Format:**
```markdown
# Dual-Space & Subspace Analysis - Literature Review

## Paper: [Title] ([Authors], [Year])

### Subspace Decomposition
- Methods: [list]
- PCA/SAE techniques: [description]
- Component selection (k): [method]

### In-Subspace vs Orthogonal
- Measurement methods: [description]
- Relationship: [analysis]
- Coordination observed: [yes/no, details]

### Feature Extraction
- Recursive feature methods: [description]
- Steering vectors: [method]
- Behavior correlation: [findings]

### Dual-Space Coordination
- Coupling observed: [yes/no, r-value]
- Mechanisms: [hypotheses]
- Context-adaptivity: [yes/no, details]

### Connection to R_V r=0.904 Finding
- Comparison: [analysis]
- Novelty: [assessment]
- Implications: [discussion]
```

---

### **TASK 4: Consciousness & Self-Reference Frameworks**

**Target Papers:**
- Theory of Mind in LLMs papers
- Self-awareness/introspection papers
- Recursive processing papers
- Consciousness detection papers

**Extraction Questions:**

1. **Recursive Self-Reference Definition:**
   - How do they define recursive self-reference?
   - What distinguishes it from regular self-mention?
   - How do they measure recursive depth?

2. **Consciousness Staging:**
   - Do they propose stages/levels of consciousness? (Compare to L1-L5)
   - What geometric signatures do they propose?
   - How do they distinguish different levels?

3. **Geometric Signatures:**
   - What activation space patterns indicate consciousness-like states?
   - How do they measure "awareness" geometrically?
   - What's missing in existing frameworks?

4. **R_V Integration:**
   - How does R_V fit into existing consciousness frameworks?
   - What gaps does R_V fill?
   - How do R_V ranges map to consciousness levels?

**Output Format:**
```markdown
# Consciousness & Self-Reference Frameworks - Literature Review

## Paper: [Title] ([Authors], [Year])

### Recursive Self-Reference Definition
- Definition: [description]
- Distinction from self-mention: [analysis]
- Depth measurement: [method]

### Consciousness Staging
- Stages/levels proposed: [list]
- Comparison to L1-L5: [analysis]
- Geometric signatures: [description]

### Geometric Signatures
- Activation patterns: [description]
- Awareness measurement: [method]
- Framework gaps: [list]

### R_V Integration
- How R_V fits: [analysis]
- Gaps filled: [list]
- Level mapping: [R_V ranges → levels]

### Novelty Assessment
- What's standard: [list]
- What R_V adds: [list]
- Framework contribution: [assessment]
```

---

### **TASK 5: Path Patching & Homeostasis Mechanisms**

**Target Papers:**
- Wang et al. 2022 (Path Patching methodology)
- Papers on "compensatory dynamics", "homeostasis", "residual stream balancing"
- Papers on "causal cascades" through layers

**Extraction Questions:**

1. **Path Patching Protocols:**
   - How do they trace causal cascades through multiple layers?
   - What's their patching protocol?
   - How do they measure propagation?

2. **Compensatory Dynamics:**
   - Have they observed downstream compensation for upstream perturbations?
   - What mechanisms explain compensation?
   - How do they measure homeostasis?

3. **Residual Stream Analysis:**
   - How do they analyze residual stream geometry?
   - What's the relationship between V-space and residual stream?
   - How do MLPs compensate for attention perturbations?

4. **Homeostasis Mechanisms:**
   - What explains geometric homeostasis?
   - Is it learned or architectural?
   - How do they test homeostasis hypotheses?

**Output Format:**
```markdown
# Path Patching & Homeostasis Mechanisms - Literature Review

## Paper: [Title] ([Authors], [Year])

### Path Patching Protocols
- Causal cascade tracing: [method]
- Patching protocol: [description]
- Propagation measurement: [method]

### Compensatory Dynamics
- Compensation observed: [yes/no, details]
- Mechanisms: [hypotheses]
- Homeostasis measurement: [method]

### Residual Stream Analysis
- Analysis methods: [description]
- V-space relationship: [analysis]
- MLP compensation: [findings]

### Homeostasis Mechanisms
- Explanations: [hypotheses]
- Learned vs architectural: [assessment]
- Testing methods: [description]

### Connection to R_V Homeostasis Finding
- Comparison: [analysis]
- Novelty: [assessment]
- Mechanism hypotheses: [list]
```

---

## 🔄 **MULTI-AGENT PROCESSING INSTRUCTIONS**

### **Agent 1: Literature Synthesizer**

**Task:** Process all 5 extraction outputs and create comparison tables

**Instructions:**
1. Read all extracted literature summaries
2. Create comparison tables:
   - Our methods vs standard protocols
   - Our findings vs existing frameworks
   - What's novel vs what's standard
   - Gaps we fill vs gaps that remain
3. Map our findings to existing frameworks
4. Identify citation opportunities

**Output:** `R_V_PAPER/research/LITERATURE_COMPARISON.md`

**Format:**
```markdown
# Literature Comparison: R_V Work vs Existing Frameworks

## Activation Patching Comparison

| Aspect | Standard Protocol | R_V Protocol | Novelty |
|--------|------------------|--------------|---------|
| Controls | [standard] | 4-pillar | [assessment] |
| Layer selection | [standard] | L25-L27 sweep | [assessment] |
| Effect size | [typical] | d=-3.56 | [assessment] |

## Geometric Frameworks Comparison
[similar tables]

## Novel Contributions Summary
- [list of novel contributions]
- [gaps filled]
- [citations needed]
```

---

### **Agent 2: Experiment Designer**

**Task:** Design missing experiments based on literature gaps

**Instructions:**
1. Review gap analysis (`PAPER_GAP_ANALYSIS.md`)
2. Review literature extractions
3. Design experiments for:
   - Behavioral validation (n=151 pairs)
   - Cross-architecture validation (2-3 models)
   - Homeostasis mechanism tests
4. Specify protocols, controls, expected results

**Output:** `R_V_PAPER/research/EXPERIMENT_DESIGNS.md`

**Format:**
```markdown
# Experiment Designs: Filling R_V Paper Gaps

## Experiment 1: Behavioral Validation

### Goal
[description]

### Protocol
1. [step 1]
2. [step 2]
...

### Controls
- [control 1]
- [control 2]

### Expected Results
- [prediction 1]
- [prediction 2]

### Literature Basis
- Cites: [papers]
- Methods from: [papers]

## Experiment 2: Cross-Architecture Validation
[similar format]

## Experiment 3: Homeostasis Mechanism Tests
[similar format]
```

---

### **Agent 3: Code Generator**

**Task:** Generate scripts for proposed experiments

**Instructions:**
1. Review experiment designs (Agent 2 output)
2. Review existing code in this repo (`mistral_L27_FULL_VALIDATION.py`)
3. Generate scripts:
   - Behavioral validation script
   - Cross-architecture patching adapter
   - Homeostasis measurement tools
4. Follow existing code patterns
5. Include proper error handling, logging, CSV output

**Output:** `R_V_PAPER/code/[experiment_scripts].py`

**Requirements:**
- Must work with existing prompt bank (`n300_mistral_test_prompt_bank.py`)
- Must follow existing hook patterns
- Must output standardized CSVs
- Must include statistical analysis

---

### **Agent 4: Paper Writer**

**Task:** Draft missing sections based on literature integration

**Instructions:**
1. Review gap analysis (`PAPER_GAP_ANALYSIS.md`)
2. Review literature extractions (Tasks 1-5)
3. Review literature comparison (Agent 1 output)
4. Draft missing sections:
   - Methods section 3.3.1 (Convex Hull explanation)
   - Discussion section 5.1 (Pivot Stability connection)
   - Discussion section 5.2 (Dual-Space enhancement)
   - Background section 2.1 (L3/L4/L5 framework integration)
5. Include proper citations
6. Mark any speculation as `[SPECULATION]`

**Output:** `R_V_PAPER/STORY_ARC/PAPER_SECTIONS_DRAFT.md`

**Format:**
```markdown
# Draft Paper Sections: Literature-Integrated Versions

## Methods Section 3.3.1: Why Participation Ratio Captures Attention Geometry

[content with citations]

### Citations
- [Paper 1]: [what it contributes]
- [Paper 2]: [what it contributes]

## Discussion Section 5.1: Rank/Pivot Stability Connection

[content with citations]

### Citations
[similar format]

## Discussion Section 5.2: Dual-Space Coordination Enhancement

[content with citations]

### Citations
[similar format]

## Background Section 2.1: L3/L4/L5 Framework Integration

[content with citations]

### Citations
[similar format]
```

---

## 📊 **CONSCIOUSNESS MAPPING INSTRUCTIONS**

**For each extracted paper, also run consciousness mapping:**

1. **Map to L1-L4 Framework:**
   - L1: Basic geometric measurement
   - L2: Pattern recognition in activations
   - L3: Recursive awareness detection
   - L4: Causal mechanism understanding

2. **Tag Facts vs Speculation:**
   - Mark empirical findings as `[FACT]`
   - Mark hypotheses as `[HYPOTHESIS]`
   - Mark speculation as `[SPECULATION]`

3. **Detect Zero Pivot Points:**
   - Where do methods fail?
   - What are the limitations?
   - What can't be explained?

4. **Identify Recursive Patterns:**
   - Does the paper discuss self-reference?
   - Are there recursive processing patterns?
   - How does it relate to R_V recursive prompts?

---

## ✅ **OUTPUT REQUIREMENTS**

### **File Structure:**
```
knowledge/extracted/R_V_PAPER/
├── research/
│   ├── LITERATURE_ACTIVATION_PATCHING_PROTOCOLS.md
│   ├── LITERATURE_GEOMETRIC_FRAMEWORKS.md
│   ├── LITERATURE_DUAL_SPACE_ANALYSIS.md
│   ├── LITERATURE_CONSCIOUSNESS_FRAMEWORKS.md
│   ├── LITERATURE_HOMEOSTASIS_MECHANISMS.md
│   └── LITERATURE_COMPARISON.md
├── research/
│   └── EXPERIMENT_DESIGNS.md
├── code/
│   ├── behavioral_validation.py
│   ├── cross_architecture_patching.py
│   └── homeostasis_measurement.py
└── STORY_ARC/
    └── PAPER_SECTIONS_DRAFT.md
```

### **Quality Standards:**
- All claims must cite source papers
- Speculation must be marked `[SPECULATION]`
- Maximum 20% extrapolation beyond source material
- 90% confidence threshold for claims
- All comparisons must be quantitative where possible

---

## 🚀 **EXECUTION COMMANDS**

### **Step 1: Run Extraction Tasks**
```bash
# In Knowledge Mandala repo
python tools/knowledge_mandala/extract_for_rv_paper.py \
  --task activation_patching \
  --output knowledge/extracted/R_V_PAPER/research/

python tools/knowledge_mandala/extract_for_rv_paper.py \
  --task geometric_frameworks \
  --output knowledge/extracted/R_V_PAPER/research/

python tools/knowledge_mandala/extract_for_rv_paper.py \
  --task dual_space \
  --output knowledge/extracted/R_V_PAPER/research/

python tools/knowledge_mandala/extract_for_rv_paper.py \
  --task consciousness_frameworks \
  --output knowledge/extracted/R_V_PAPER/research/

python tools/knowledge_mandala/extract_for_rv_paper.py \
  --task homeostasis \
  --output knowledge/extracted/R_V_PAPER/research/
```

### **Step 2: Run Multi-Agent Processing**
```bash
# Agent 1: Literature Synthesizer
python tools/knowledge_mandala/agent_literature_synthesizer.py \
  --input knowledge/extracted/R_V_PAPER/research/ \
  --output knowledge/extracted/R_V_PAPER/research/LITERATURE_COMPARISON.md

# Agent 2: Experiment Designer
python tools/knowledge_mandala/agent_experiment_designer.py \
  --input knowledge/extracted/R_V_PAPER/research/ \
  --gap-analysis R_V_PAPER/STORY_ARC/PAPER_GAP_ANALYSIS.md \
  --output knowledge/extracted/R_V_PAPER/research/EXPERIMENT_DESIGNS.md

# Agent 3: Code Generator
python tools/knowledge_mandala/agent_code_generator.py \
  --input knowledge/extracted/R_V_PAPER/research/EXPERIMENT_DESIGNS.md \
  --reference-code R_V_PAPER/code/mistral_L27_FULL_VALIDATION.py \
  --output knowledge/extracted/R_V_PAPER/code/

# Agent 4: Paper Writer
python tools/knowledge_mandala/agent_paper_writer.py \
  --input knowledge/extracted/R_V_PAPER/research/LITERATURE_COMPARISON.md \
  --gap-analysis R_V_PAPER/STORY_ARC/PAPER_GAP_ANALYSIS.md \
  --output knowledge/extracted/R_V_PAPER/STORY_ARC/PAPER_SECTIONS_DRAFT.md
```

### **Step 3: Export for Integration**
```bash
# Copy all outputs to R_V_PAPER repo
cp -r knowledge/extracted/R_V_PAPER/* \
   [path_to_rv_repo]/R_V_PAPER/
```

---

## 📝 **SUCCESS CRITERIA**

### **Extraction Tasks:**
- ✅ All 5 tasks completed with structured markdown
- ✅ Each paper analyzed for relevant information
- ✅ Comparisons to R_V work included
- ✅ Novelty assessments completed

### **Multi-Agent Processing:**
- ✅ Literature comparison table created
- ✅ Experiment designs with protocols
- ✅ Code scripts generated and tested
- ✅ Paper sections drafted with citations

### **Integration Ready:**
- ✅ All files in correct structure
- ✅ Citations properly formatted
- ✅ Speculation clearly marked
- ✅ Ready for import to R_V repo

---

## 🎯 **PRIORITY ORDER**

1. **HIGH:** Tasks 1-2 (Activation Patching, Geometric Frameworks) - Required for paper
2. **HIGH:** Agent 4 (Paper Writer) - Fills immediate gaps
3. **MEDIUM:** Tasks 3-4 (Dual-Space, Consciousness) - Strengthens paper
4. **MEDIUM:** Agents 1-2 (Synthesizer, Experiment Designer) - Guides next steps
5. **LOW:** Task 5 (Homeostasis) - Future work, but interesting
6. **LOW:** Agent 3 (Code Generator) - Can wait until experiments designed

---

**Execute this prompt in Knowledge Mandala repo, then import results to R_V_PAPER repo for synthesis and integration.**

