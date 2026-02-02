# MECHANISTIC INTERPRETABILITY KNOWLEDGE BASE

*Generated: 2025-11-16 19:31:52*

This file consolidates key papers, blogs, and code references for MI research.
Reference this file in Cursor using `@mech_interp_knowledge_base.md`

---

## TABLE OF CONTENTS

- [Key Papers](#key-papers)
- [Essential Blogs](#essential-blogs)
- [Code Resources](#code-resources)
- [Quick Reference: Activation Patching](#quick-reference-activation-patching)
- [Key Techniques](#key-techniques)
- [Statistical Best Practices](#statistical-best-practices)
- [Common Pitfalls](#common-pitfalls)

---

## KEY PAPERS

### 1. Locating and Editing Factual Associations in GPT

**Authors:** Meng et al. 2022  
**Link:** [https://arxiv.org/abs/2202.05262](https://arxiv.org/abs/2202.05262)  
**Why Relevant:** Foundational activation patching paper - defines clean/corrupt runs, causal mediation  

**Abstract:**

> We analyze the storage and recall of factual associations in autoregressive transformer language models, finding evidence that these associations correspond to localized, directly-editable computations. We first develop a causal intervention for identifying neuron activations that are decisive in a model's factual predictions. This reveals a distinct set of steps in middle-layer feed-forward modules that mediate factual predictions while processing subject tokens. To test our hypothesis that ...

**Key Sections:** Abstract, Section 2: Causal Tracing, Section 3: Methods, Section 4: Controls

---

### 2. Interpretability in the Wild: Circuit for Indirect Object Identification

**Authors:** Wang et al. 2022  
**Link:** [https://arxiv.org/abs/2211.00593](https://arxiv.org/abs/2211.00593)  
**Why Relevant:** Systematic patching protocols and controls, path patching methodology  

**Abstract:**

> Research in mechanistic interpretability seeks to explain behaviors of machine learning models in terms of their internal components. However, most previous work either focuses on simple behaviors in small models, or describes complicated behaviors in larger models with broad strokes. In this work, we bridge this gap by presenting an explanation for how GPT-2 small performs a natural language task called indirect object identification (IOI). Our explanation encompasses 26 attention heads grou...

**Key Sections:** Section 3: Activation Patching, Section 4: Systematic Ablations

---

### 3. Does Localization Inform Editing?

**Authors:** Hase et al. 2023  
**Link:** [https://arxiv.org/abs/2301.04213](https://arxiv.org/abs/2301.04213)  
**Why Relevant:** Critical analysis of patching limitations, norm-matching requirements  

**Abstract:**

> Language models learn a great quantity of factual information during pretraining, and recent work localizes this information to specific model weights like mid-layer MLP weights. In this paper, we find that we can change how a fact is stored in a model by editing weights that are in a different location than where existing methods suggest that the fact is stored. This is surprising because we would expect that localizing facts to specific model parameters would tell us where to manipulate kno...

**Key Sections:** Section 2.1: Methodology, Section 3: Pitfalls

---

### 4. A Mathematical Framework for Transformer Circuits

**Authors:** Elhage et al. 2021  
**Link:** [https://transformer-circuits.pub/2021/framework/index.html](https://transformer-circuits.pub/2021/framework/index.html)  
**Why Relevant:** Core transformer architecture understanding for mechanistic analysis  

**Key Sections:** Residual Stream, Attention Patterns, QKV Matrices

---

### 5. In-context Learning and Induction Heads

**Authors:** Olsson et al. 2022  
**Link:** [https://arxiv.org/abs/2209.11895](https://arxiv.org/abs/2209.11895)  
**Why Relevant:** How transformers learn from context, emergence of capabilities  

**Abstract:**

> "Induction heads" are attention heads that implement a simple algorithm to complete token sequences like [A][B] ... [A] -&gt; [B]. In this work, we present preliminary and indirect evidence for a hypothesis that induction heads might constitute the mechanism for the majority of all "in-context learning" in large transformer models (i.e. decreasing loss at increasing token indices). We find that induction heads develop at precisely the same point as a sudden sharp increase in in-context learni...

**Key Sections:** Abstract, Induction Head Mechanism, Phase Transitions

---

### 6. Toy Models of Superposition

**Authors:** Elhage et al. 2022  
**Link:** [https://transformer-circuits.pub/2022/toy_model/index.html](https://transformer-circuits.pub/2022/toy_model/index.html)  
**Why Relevant:** Why features interfere in neural networks, relevant for V-space geometry  

**Key Sections:** Superposition Hypothesis, Interference Patterns

---

### 7. Towards Monosemanticity: Decomposing Language Models With Dictionary Learning

**Authors:** Anthropic 2023  
**Link:** [https://transformer-circuits.pub/2023/monosemantic-features/index.html](https://transformer-circuits.pub/2023/monosemantic-features/index.html)  
**Why Relevant:** Disentangling superposed features using SAEs  

**Key Sections:** Sparse Autoencoders, Feature Dictionaries

---

### 8. Causal Scrubbing: A Method for Rigorously Testing Interpretability Hypotheses

**Authors:** Chan et al. 2022  
**Link:** [https://arxiv.org/abs/2301.04785](https://arxiv.org/abs/2301.04785)  
**Why Relevant:** Higher standard for causal claims than activation patching  

**Abstract:**

> Adversarial training has been considered an imperative component for safely deploying neural network-based applications to the real world. To achieve stronger robustness, existing methods primarily focus on how to generate strong attacks by increasing the number of update steps, regularizing the models with the smoothed loss function, and injecting the randomness into the attack. Instead, we analyze the behavior of adversarial training through the lens of response frequency. We empirically di...

**Key Sections:** Section 2: Causal Scrubbing Algorithm, Section 3: Examples

---

### 9. Representation Engineering: A Top-Down Approach to AI Transparency

**Authors:** Zou et al. 2023  
**Link:** [https://arxiv.org/abs/2310.01405](https://arxiv.org/abs/2310.01405)  
**Why Relevant:** Geometric view of representations, relevant for V-space analysis  

**Abstract:**

> In this paper, we identify and characterize the emerging area of representation engineering (RepE), an approach to enhancing the transparency of AI systems that draws on insights from cognitive neuroscience. RepE places population-level representations, rather than neurons or circuits, at the center of analysis, equipping us with novel methods for monitoring and manipulating high-level cognitive phenomena in deep neural networks (DNNs). We provide baselines and an initial analysis of RepE tec...

**Key Sections:** Section 3: Reading Vectors, Section 4: Control Vectors

---

### 10. Finding Neurons in a Haystack: Case Studies with Sparse Probing

**Authors:** Gurnee et al. 2023  
**Link:** [https://arxiv.org/abs/2305.01610](https://arxiv.org/abs/2305.01610)  
**Why Relevant:** Methods for finding sparse, interpretable features  

**Abstract:**

> Despite rapid adoption and deployment of large language models (LLMs), the internal computations of these models remain opaque and poorly understood. In this work, we seek to understand how high-level human-interpretable features are represented within the internal neuron activations of LLMs. We train $k$-sparse linear classifiers (probes) on these internal activations to predict the presence of features in the input; by varying the value of $k$ we study the sparsity of learned representation...

**Key Sections:** Sparse Linear Probing, Neuron-Level Interpretability

---

## ESSENTIAL BLOGS

### 1. A Comprehensive Mechanistic Interpretability Explainer & Glossary

**Author:** Neel Nanda  
**Link:** [https://www.neelnanda.io/mechanistic-interpretability/glossary](https://www.neelnanda.io/mechanistic-interpretability/glossary)  
**Why Relevant:** Best practical guide to MI techniques with code examples  
**Key Topics:** Activation Patching, Causal Tracing, Residual Stream, Hook Points

---

### 2. 200 Concrete Open Problems in Mechanistic Interpretability

**Author:** Neel Nanda  
**Link:** [https://www.neelnanda.io/mechanistic-interpretability/problems](https://www.neelnanda.io/mechanistic-interpretability/problems)  
**Why Relevant:** Identify gaps and opportunities in current MI research  
**Key Topics:** Research Directions, Unsolved Questions, Difficulty Ratings

---

### 3. An Extremely Opinionated Annotated List of MI Papers

**Author:** Neel Nanda  
**Link:** [https://www.neelnanda.io/mechanistic-interpretability/papers](https://www.neelnanda.io/mechanistic-interpretability/papers)  
**Why Relevant:** Curated reading list with practical commentary  
**Key Topics:** Paper Reviews, Reading Order, Key Insights

---

### 4. Causal Scrubbing: A Method for Rigorously Testing Interpretability

**Author:** Redwood Research  
**Link:** [https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing)  
**Why Relevant:** Higher bar for causal claims than patching  
**Key Topics:** Rigorous Causality, Hypothesis Testing, Computational Graphs

---

### 5. The Singular Value Decompositions of Transformer Weight Matrices are Highly Interpretable

**Author:** Anthropic  
**Link:** [https://transformer-circuits.pub/2024/svd-interp/index.html](https://transformer-circuits.pub/2024/svd-interp/index.html)  
**Why Relevant:** Directly relevant to your R_V metric using SVD  
**Key Topics:** SVD Analysis, Weight Decomposition, Geometric Structure

---

## CODE RESOURCES

### 1. TransformerLens Documentation

**Link:** [https://transformerlens.org/](https://transformerlens.org/)  
**Why Relevant:** Standard library for MI research with HuggingFace integration  
**Key Sections:** Hook Points, Activation Patching Examples, Cache System

---

### 2. TransformerLens GitHub

**Link:** [https://github.com/neelnanda-io/TransformerLens](https://github.com/neelnanda-io/TransformerLens)  
**Why Relevant:** Working code examples and utilities  
**Key Files:** demos/Activation_Patching_in_TL_Demo.ipynb, demos/Exploratory_Analysis_Demo.ipynb

---

### 3. CircuitsVis - Interactive Attention Visualizations

**Link:** [https://github.com/alan-cooney/CircuitsVis](https://github.com/alan-cooney/CircuitsVis)  
**Why Relevant:** Visualization tools for understanding attention and activations  
**Key Sections:** Attention Patterns, Neuron Activations

---

### 4. SAELens - Sparse Autoencoder Training

**Link:** [https://github.com/jbloomAus/SAELens](https://github.com/jbloomAus/SAELens)  
**Why Relevant:** Tools for training sparse autoencoders on language models  
**Key Sections:** Training SAEs, Feature Extraction

---

## QUICK REFERENCE: ACTIVATION PATCHING

### The Core Logic (from Meng et al. 2022)

```
Setup:
  Clean run:    Prompt A → Layer L → Activation X → Output Y
  Corrupt run:  Prompt B → Layer L → Activation Z → Output W

Intervention:
  Patched run:  Prompt B → Layer L → [Replace with X] → Output Y'

Causal claim:
  If Y' ≈ Y (not W), then:
  → Layer L causally mediates the effect
  → Activation X contains the critical information
```

### Implementation Pattern (PyTorch)

```python
from contextlib import contextmanager

@contextmanager
def patch_activation(model, layer_idx, source_activation):
    """Hook to replace activations at specified layer"""
    def hook_fn(module, input, output):
        # Replace output with source activation
        return source_activation
    
    handle = model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()
```

### Required Control Conditions

1. **Random patch:** Replace with Gaussian noise (norm-matched) → Tests if ANY change is sufficient
2. **Shuffled patch:** Permute source activation → Tests if structure is necessary
3. **Wrong-layer patch:** Patch at different depth → Tests if this layer is special
4. **Opposite-direction patch:** Use opposite-type source → Tests directionality
5. **Partial patch:** Only patch subset of neurons → Tests distributed vs localized

---

## KEY TECHNIQUES

### Activation Patching

**Description:** Replace activations at layer L from run A with those from run B  
**When to use:** Testing if layer L causally mediates an effect  
**Controls needed:** Random noise patch, Shuffled patch, Wrong-layer patch, Opposite-direction patch  

### Causal Tracing

**Description:** Systematically restore clean activations to corrupted run  
**When to use:** Localizing where information is processed  
**Controls needed:** Multiple corruption types, Gradual restoration  

### Path Patching

**Description:** Patch specific attention head outputs or MLPs  
**When to use:** Testing specific circuit components  
**Controls needed:** Full model ablation, Random subset patching  

### Logit Lens

**Description:** Decode intermediate representations to vocabulary  
**When to use:** Understanding what information is present at each layer  
**Controls needed:** Layer normalization, Unembedding matrix choice  

### Attention Pattern Analysis

**Description:** Analyze where attention heads look  
**When to use:** Understanding information flow  
**Controls needed:** Positional vs semantic attention, Head importance weighting  

---

## STATISTICAL BEST PRACTICES

### Sample Size Calculation

For activation patching with expected effect size d:
- Small effect (d=0.2): n ≈ 200 pairs
- Medium effect (d=0.5): n ≈ 50 pairs
- Large effect (d=0.8): n ≈ 20 pairs

Add 20% for multiple comparisons correction.

### Statistical Tests

1. **Within-pair differences:** Paired t-test or Wilcoxon signed-rank
2. **Multiple conditions:** ANOVA with post-hoc tests
3. **Multiple comparisons:** Bonferroni (conservative) or Benjamini-Hochberg FDR
4. **Effect size:** Cohen's d for magnitude (not just significance)

### Pre-registration

Before running experiments, specify:
- Primary hypothesis
- Sample size
- Statistical threshold (e.g., p < 0.01)
- Exclusion criteria
- Analysis plan

---

## COMMON PITFALLS

### From Hase et al. 2023 & Community Experience

1. **Localization ≠ Sufficiency** - Just because a layer shows an effect doesn't mean it's the only place
2. **Norm Matching** - Random patches MUST match the norm of real patches to avoid trivial effects
3. **Multiple Comparison Correction** - Use Bonferroni or FDR when testing multiple conditions
4. **Distribution Shift** - Patching can create impossible activation patterns
5. **Prompt Length Bias** - Ensure balanced prompt lengths between conditions
6. **Layer Normalization** - Be careful with patching before/after LayerNorm
7. **Batch Effects** - Run conditions in randomized order, not blocks
8. **Numerical Stability** - SVD can fail on degenerate matrices (add small epsilon)

### Debugging Checklist

- [ ] Hooks properly removed after use
- [ ] Gradients disabled during inference (model.eval())
- [ ] Device placement consistent (.to(device))
- [ ] Memory cleared between runs (torch.cuda.empty_cache())
- [ ] Random seeds set for reproducibility

---

## PROJECT-SPECIFIC: R_V METRIC

### Definition

```
R_V(layer) = PR(V_layer) / PR(V_early)

Where:
  PR = Participation Ratio = (Σλᵢ)² / Σλᵢ²
  λᵢ = singular values from SVD of V matrix
  V = value projection outputs (last window_size tokens)
```

### Interpretation

- R_V < 1.0: Geometric contraction (reduced effective rank)
- R_V ≈ 1.0: Neutral (no change in geometry)
- R_V > 1.0: Geometric expansion (increased effective rank)

### Key Findings

- Recursive prompts show 15-24% contraction at layer ~27/32
- Effect is universal across architectures (Mistral, Qwen, Llama, etc.)
- MoE models show stronger effect (24% vs 15% for dense)

---

*End of Knowledge Base*
