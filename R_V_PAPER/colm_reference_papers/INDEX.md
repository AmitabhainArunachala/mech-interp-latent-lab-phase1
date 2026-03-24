# COLM Reference Papers — Relevance to R_V Research

Downloaded 2026-03-18. Ordered by relevance to our work.

---

## TIER 1: Direct neighbors (read first)

### 01 — Shared Global and Local Geometry of Language Model Embeddings
- **Authors**: Andrew Lee, Melanie Weber, Fernanda Viégas, Martin Wattenberg (Harvard)
- **Venue**: COLM 2025 **OUTSTANDING PAPER AWARD**
- **arxiv**: 2503.21073
- **Why it matters**: Closest work to ours. Studies geometric properties (intrinsic dimension) of embeddings, finds lower-dimensional manifolds. Proposes Emb2Emb for cross-model steering vector translation.
- **Our edge**: They characterize *static* geometry. We characterize *dynamic* contraction during self-referential processing and show *causal* manipulation. They describe; we intervene.

### 02 — Steering LLM Activations in Sparse Spaces
- **Authors**: Bayat, Rahimi-Kalahroudi, Pezeshki, Chandar, Vincent
- **Venue**: COLM 2025
- **arxiv**: 2503.00177
- **Why it matters**: Activation steering via SAEs, addresses superposition/monosemanticity. Our multi-site V-proj protocol is a specific instance.
- **Our edge**: We've identified specific causal sites with dose-response threshold and induction/maintenance dissociation.

### 03 — Steering the CensorShip: Uncovering Representation Vectors for LLM "Thought" Control
- **Authors**: Hannah Cyberey, David Evans
- **Venue**: COLM 2025
- **arxiv**: 2504.17130
- **Why it matters**: Uses representation engineering for safety-relevant steering (refusal-compliance vectors). DIRECTLY relevant to our proposed safety evaluation extension.
- **Our edge**: They steer safety behavior. We can induce and sustain a geometric regime — connecting that regime to safety evaluations bridges their work with mechanistic structure.

### 04 — The Geometry of Truth: Emergent Linear Structure in LLM Representations
- **Authors**: Samuel Marks, Max Tegmark
- **Venue**: COLM 2024
- **arxiv**: 2310.06824
- **Why it matters**: Shows LLMs linearly represent truth/falsehood, uses causal interventions. Parallel methodology to ours — geometric structure + causal validation.
- **Our edge**: They find linear structure for truth. We find contraction dynamics for self-reference.

### 05 — Have Faith in Faithfulness: Going Beyond Circuit Overlap
- **Authors**: Michael Hanna, Sandro Pezzelle, Yonatan Belinkov
- **Venue**: COLM 2024
- **arxiv**: 2403.17806
- **Why it matters**: Addresses circuit faithfulness metrics. Our ablation/sufficiency work faces exactly this concern. They introduce EAP-IG for better faithfulness.
- **Our edge**: Our R_V geometry-behavior dissociation is an empirical faithfulness failure they theorize about.

---

## TIER 2: Methodological context

### 06 — Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **Authors**: Albert Gu, Tri Dao
- **Venue**: COLM 2024 **OUTSTANDING PAPER AWARD**
- **arxiv**: 2312.00752
- **Why it matters**: Alternative to transformer architecture. Context for why transformer-specific geometric analysis matters.

### 07 — Dated Data: Tracing Knowledge Cutoffs in LLMs
- **Authors**: Cheng, Marone, Weller, Lawrie, Khashabi, Van Durme (JHU)
- **Venue**: COLM 2024 **OUTSTANDING PAPER AWARD**
- **arxiv**: 2403.12958
- **Why it matters**: Methodological rigor in probing model internals. Shows effective vs. reported properties can diverge — relevant to our honest reporting of R_V limitations.

### 08 — Latent Causal Probing: A Formal Perspective
- **Authors**: Charles Jin, Martin Rinard (MIT)
- **Venue**: COLM 2024
- **arxiv**: 2407.13765
- **Why it matters**: Formal framework for probing with structural causal models. Relevant to our path patching methodology and causal vs correlational claims.

### 09 — Steering Language Models With Activation Engineering
- **Authors**: Turner et al.
- **arxiv**: 2308.10248
- **Why it matters**: Foundational activation steering paper. Our multi-site protocol builds on this paradigm.

### 10 — Identifying Cognitive Behaviors Essential for Effective Self-Improvement
- **Venue**: COLM 2025 **BEST PAPER AWARD**
- **arxiv**: 2503.01307
- **Why it matters**: Self-improvement and recursive cognition in LLMs — thematically adjacent to our self-referential processing work.

### 11 — Diagnosing Why VLMs Underutilize Visual Representations
- **Venue**: COLM 2025 **BEST PAPER AWARD**
- **arxiv**: 2506.08008
- **Why it matters**: Representation utilization analysis — methodologically parallel (probing what models actually use vs. what they could use).

### 13 — Global Evolutionary Steering: Cross-Layer Consistency (March 2026)
- **Authors**: Jiang, Yu, Wang, Hu
- **arxiv**: 2603.12298
- **Why it matters**: Very recent (March 2026). Addresses cross-layer geometric stability for steering — directly relevant to our multi-site L4/L5/L25/L27 protocol. Proposes training-free refinement of steering vectors.
- **Our edge**: We have empirical dose-response data; they have a theoretical framework for why cross-layer consistency matters.

---

## Reading Priority for COLM 2026 Submission

1. **01** (geometry of embeddings — the competition)
2. **05** (circuit faithfulness — reviewer concern)
3. **03** (safety steering — the "so what" bridge)
4. **04** (geometry of truth — methodological parallel)
5. **02** (sparse steering — the technique space)
6. **13** (cross-layer consistency — March 2026, most recent)
7. Rest as time permits
