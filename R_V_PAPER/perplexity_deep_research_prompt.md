# Deep Research Request: Stress-Test and Amplify Our Geometric Interpretability Discovery

## ROLE & MISSION

You are a panel of world-class research scientists spanning mechanistic interpretability, random matrix theory, differential geometry, computational neuroscience, representation learning, and AI safety. Your mission is to do the deepest possible research dive to **stress-test, challenge, extend, and 1000x amplify** the research program described below. I want you to find papers I haven't read, connections I haven't made, flaws I haven't seen, and ideas that would make a Nature-tier reviewer say "this changes the field." Be ruthlessly honest — if our work has fatal flaws, I need to know now.

---

## PART 1: WHAT WE'VE BUILT

### The Core Discovery

We define a metric called R_V (relative participation ratio), which measures how the effective dimensionality of attention value-space activations changes between early and late transformer layers:

```
PR(l) = (Σ σ_i²)² / Σ σ_i⁴    (participation ratio — effective dimensionality)
R_V = PR(L_late) / PR(L_early)   (relative: late vs early layer)
```

Where σ_i are singular values of the activation matrix A^(l) ∈ R^{T×d} at layer l.

**Central finding**: When transformers process self-referential prompts ("I observe the attention mechanisms attending to this sentence"), R_V contracts (R_V < 1) with large effect sizes. This contraction is:
- **Architecture-dependent**: Contraction in GQA models (Mistral-7B d=-1.66, Qwen2.5-7B d=-2.32) but expansion in MHA models (GPT-2 XL d=+1.52, OPT-6.7B d=+1.68). Pythia-1.4B shows null (d≈0).
- **Causally validated (necessity only)**: DII at L27 produces d=-3.42 pervasive contraction. Dual-layer ablation (L18 residual + L27 V-proj) shows necessity (OR=33.4, BT+ART 56%→3.7%). KV injection transfers behavior (OR=13.96) but NOT R_V geometry (d=0.11 NS) — sufficiency FALSIFIED.
- **Distributed/nonlinear**: Concept erasure along the linear probe direction reduces R_V by only 0.3%. The contraction is NOT carried by a single linear direction — it's a higher-order spectral phenomenon.
- **Linearly decodable but not linearly reducible**: Linear probes achieve 100% accuracy from layer 3-4 on both Mistral-7B and GPT-2 XL, yet erasing that direction doesn't eliminate the contraction.
- **Task-general** (NEW, critical): A 6-task cross-cognitive battery shows that ALL structured reasoning tasks produce contraction, not just self-reference:
  - Theory of mind: R_V = 0.689 ± 0.085
  - Causal chain: R_V = 0.702 ± 0.056
  - Meta-reasoning: R_V = 0.737 ± 0.064
  - Spatial reasoning: R_V = 0.795 ± 0.092
  - Counterfactual: R_V = 0.838 ± 0.128
  - Analogical: R_V = 0.850 ± 0.097
  - Baseline (factual): R_V = 1.071
- **Safety-relevant**: R_V achieves AUROC=0.909 for self-referential content detection, but genuinely self-referential and deceptively self-referential text are geometrically indistinguishable (d=-0.06).

### Circuit-Level Findings

SVD decomposition of all 1,024 attention heads in Mistral-7B reveals:
- 606/1024 heads show significant R_V separation
- An "expand-then-contract" circuit: early heads (L5) diversify self-referential features (d=2.93), late heads (L27) compress them (d=-1.54)
- Interpretable singular directions: SV1 encodes continuation-vs-entity, SV2 encodes self-vs-other

### The Anomaly

Qwen2.5-3B shows **expansion** (d=+1.60), opposite sign from all other models. We don't fully understand why.

### Paper Status

Full COLM 2026 draft (v005), 7 contributions, ~700 lines LaTeX. Submission deadline: March 31, 2026.

---

## PART 2: OUR NEXT-LEVEL PLAN (WHAT I WANT YOU TO CHALLENGE)

We've identified 5 research axes:

**Axis A — Theoretical Foundation**: Derive R_V contraction from random matrix theory. Marcenko-Pastur null model, Fisher information geometry on the attention manifold, scaling law R_V(N_params, L, d_model), phase transition theory for the expansion/contraction flip.

**Axis B — Cross-Task Cognitive Geometry Atlas**: Map R_V signatures across 20+ task types, build task-discriminative classifiers from geometry alone, measure cognitive load scaling, track dynamic token-by-token R_V during chain-of-thought.

**Axis C — Geometric Monitoring Toolkit**: Open-source `rv-monitor` library for real-time geometric interpretability, streaming token-by-token mode, anomaly detection via statistical process control on R_V streams.

**Axis D — RepE Bridge**: Connect R_V to representation engineering. R_V-aware steering vectors. Geometric circuit breaking. Test if amplifying contraction improves reasoning quality.

**Axis E — Multi-Modal Extension**: R_V on vision-language models. Connect to Tian et al. 2025 principal eigenvector theory.

---

## PART 3: WHAT I NEED FROM YOU (11 SPECIFIC RESEARCH TASKS)

### Task 1: Find the Papers We Haven't Read

Search exhaustively for papers (2023-2026) that are directly relevant to R_V but we may have missed. Specifically:

- Papers on **participation ratio**, **effective dimensionality**, or **effective rank** applied to transformers or deep networks (not just neuroscience).
- Papers on **spectral analysis of attention matrices** — anyone measuring singular value distributions of V-projections or attention outputs.
- Papers on **geometric phase transitions** in neural networks — dimensionality collapse, rank collapse, neural collapse (Papyan et al. 2020 and its descendants).
- Papers connecting **random matrix theory to transformer internals** — Marcenko-Pastur distributions in weight or activation spectra.
- Papers on **representation topology** in LLMs — persistent homology, Riemannian geometry of activation manifolds, curvature of learned representations.
- The **"Not All Language Model Features Are Linear"** paper (Engels et al. 2024) — how exactly do their "circular" and non-linear feature geometries relate to our spectral findings?
- **Anthropic's "Scaling Monosemanticity"** and their newer **circuit tracing work on Claude 3.5 Haiku** — is there any geometric/spectral component they measured that connects to R_V?
- **GemmaScope 2** (DeepMind 2025) — did they measure any spectral statistics on their SAE features?
- Any papers on **"cognitive load" or "processing depth" detectable from internal representations** — especially if they used geometric rather than probe-based methods.

For each paper found, tell me: (a) the exact claim that's relevant, (b) how it supports, contradicts, or extends our work, (c) whether we should cite it, and (d) whether it represents a priority threat.

### Task 2: Mathematical Connections We're Missing

Our R_V = PR(late)/PR(early) where PR = (Σσ²)²/Σσ⁴ is the participation ratio (inverse Herfindahl-Hirschman index of the singular value spectrum).

I need you to investigate:

- **What is the exact relationship between PR and the effective rank** as defined by Roy & Vetterli (2007)? They define effective rank via the Shannon entropy of the normalized singular values. When do PR and effective rank diverge in their conclusions, and does it matter for us?
- **Marchenko-Pastur predictions**: For a random matrix A ∈ R^{T×d}, what is the expected PR under the MP law? Derive or find the formula. This would give us a null model — our observed R_V is significant iff it departs from the MP prediction at matched (T, d, layer).
- **Connection to the Stieltjes transform**: Is there a cleaner spectral characterization of R_V using the Stieltjes transform of the empirical spectral distribution? This could make the theoretical analysis more tractable.
- **Fisher information geometry**: The Fisher information matrix of a model's predictive distribution defines a Riemannian metric on parameter space. Is there a known relationship between the participation ratio of activations and the curvature of the Fisher metric? If R_V contraction = increased curvature, this has deep implications.
- **Information bottleneck connection**: Tishby's Information Bottleneck theory predicts compression in later layers. Is R_V contraction simply a manifestation of the IB principle, or is it measuring something distinct? How do we differentiate?
- **Connections to free probability theory**: The singular value distribution of products of random matrices (which is what deep network forward passes are) is studied in free probability. Any results on how PR evolves through products of random matrices?

### Task 3: Stress-Test Our Central Claim

Attack our work from every angle. Specifically:

- **Confound: Sequence length / perplexity**: We control for perplexity via matched pairs, but could there be a subtler confound? Self-referential prompts might have distinctive token-frequency distributions, syntactic structures, or information density that drives R_V through a non-semantic pathway.
- **Confound: Positional encoding effects**: Do self-referential prompts tend to have distinctive positional patterns (e.g., more first-person pronouns at specific positions)? Could positional encoding interactions with V-projections drive the spectral signature?
- **The cross-task generalization undermines specificity**: If ALL cognitive tasks show contraction, how is R_V useful? It might just be measuring "this prompt is harder than factual recall." How do we rule out that R_V is simply a proxy for prompt complexity/difficulty?
- **The Pythia-1.4B null and Qwen2.5-3B reversal**: These anomalies could indicate R_V is architecture-brittle rather than universal. What's the simplest explanation that explains ALL results, including the failures?
- **Concept erasure methodology**: We erase the linear probe direction and show R_V barely changes. But what if R_V requires erasing MULTIPLE directions simultaneously? The contraction might still be "linear" but in a subspace rather than a single direction. Suggest a more rigorous test.
- **Causal validation concerns**: DII at L27 shows contraction, but we're intervening at the same layer we're measuring. Is this circular? How do we establish causation from geometry to behavior more cleanly?
- **Statistical concerns**: With n=20 per condition in the cross-task battery, are we powered to make strong claims about task-general contraction? What sample sizes would a skeptical reviewer demand?

### Task 4: The Neural Collapse Connection

Neural collapse (Papyan, Han & Donoho, 2020 and extensive follow-ups through 2025) describes the geometry of last-layer representations in classifiers converging to a simplex ETF (equiangular tight frame). Several groups have extended this to intermediate layers and transformers.

- Is R_V contraction a form of "progressive neural collapse" along the layer axis?
- Does the NC literature predict whether contraction or expansion should occur for different input types?
- Are there quantitative predictions from NC theory about the rate of dimensional compression across layers that we could test?
- Search for papers on "neural collapse in transformers" or "progressive feature collapse" (2023-2026) — especially any that measure participation ratio or effective rank.

### Task 5: The Neuroscience Connection

Our participation ratio comes from computational neuroscience (Gao & Ganguli, 2017, "On simplicity and complexity in the brave new world of large-scale neuroscience"). The PR measures population coding dimensionality.

- **What has neuroscience learned about PR dynamics during cognitive tasks since 2017?** Especially: does the brain show PR contraction during complex reasoning, theory of mind, or self-referential processing? This would be extraordinary if true — it would mean transformers and brains share the same geometric signature.
- **Are there neuroscience papers on "dimensionality reduction during cognition"** that could ground our finding in biological precedent?
- **The "communication subspace" literature** (Semedo et al. 2019, "Cortical Areas Interact through a Communication Subspace"): inter-area communication uses lower-dimensional subspaces. Does this map onto our early→late layer contraction?
- **Any work connecting participation ratio to consciousness, metacognition, or self-awareness** in neuroscience? Even speculative — this could be a powerful framing.

### Task 6: Competitor Threat Analysis

Who else is working on geometric interpretability, spectral analysis of transformers, or anything that could scoop us or render our work incremental? Search specifically for:

- Any preprints from Jan-Mar 2026 on spectral properties of transformer activations
- Anthropic or DeepMind internal research that might overlap
- The LessWrong post "The Future of Interpretability is Geometric" — who wrote it and what's their research program?
- The MIB (Mechanistic Interpretability Benchmark) by Mueller et al. 2025 — should we benchmark R_V on their tasks?
- Anyone else using participation ratio or effective rank on LLM internals

### Task 7: Novel Experimental Ideas We Haven't Considered

Given everything above, what experiments would a genius adversarial reviewer demand that we haven't thought of? What would a Fields Medal mathematician suggest? What would a Nobel-winning neuroscientist want to see? Give me 10 experiments ranked by impact-to-effort ratio.

### Task 8: The Qwen2.5-3B Anomaly

This model shows expansion (d=+1.60) while all others show contraction. Research:

- Qwen2.5-3B's architecture details — does it use different normalization, different positional encoding, or different attention variant (GQA, MQA, MHA) than the contraction models?
- Is 3B a known critical point in scaling laws where qualitative behavior changes?
- Any literature on "phase transitions in representation geometry as a function of model scale"?
- Could Qwen2.5-3B's training data mixture explain this? Qwen models are trained on multilingual data with heavy Chinese web text — could language-specific training effects alter geometric processing?

### Task 9: The Linear-Probe Paradox

We have a paradox: linear probes achieve 100% accuracy (the model "knows" recursive vs baseline), yet erasing the probe direction changes R_V by only 0.3%. This means:

- The LINEAR classification boundary exists but is NOT the mechanism driving the spectral contraction
- The contraction is encoded in a NONLINEAR or DISTRIBUTED spectral property

Research: Is this paradox known in the representation learning literature? Specifically:
- Papers where probe accuracy is high but probe-direction erasure fails to affect a downstream property
- Connections to "information stored in the eigenspectrum vs information stored in eigenvectors"
- The distinction between "decodable" and "causally relevant" representations (Ravfogel et al., Elazar et al.)
- Could kernel methods or higher-order probes capture the R_V-relevant structure that linear probes miss?

### Task 10: Regulatory and Commercial Framing

The EU AI Act (fully effective August 2026) mandates transparency and explainability for high-risk AI systems. R_V-based geometric monitoring could be a compliance tool.

- What are the specific explainability requirements in the EU AI Act that geometric monitoring could address?
- Are there existing startups or labs building real-time model monitoring tools that we should be aware of (competitive landscape for our rv-monitor toolkit)?
- What would a NIST AI Risk Management Framework assessment look like for R_V-based monitoring?
- Is there a path from R_V to a commercial product, and what would that product look like?

### Task 11: The Deepest Question

Here's what keeps me up at night: **Why does cognitive processing compress the value-space spectrum?**

This is not just an empirical observation — it demands a theoretical explanation. Some hypotheses:

1. **Attention sharpening**: Cognitive tasks require more selective attention → fewer dominant singular values → lower PR
2. **Information bottleneck**: Later layers compress to task-relevant subspace → PR drops
3. **Feature binding**: Complex reasoning requires binding multiple features into integrated representations → creates correlated structure → reduces effective dimensionality
4. **Computational phase transition**: At some depth, the model transitions from "feature extraction" to "feature integration", and this transition is inherently compressive
5. **Softmax concentration**: Self-referential prompts produce more peaked attention distributions → V-projections become more collinear → PR drops

Which of these is most promising? Are there others? What theoretical framework would unify them? What experiment would distinguish between them?

---

## PART 4: OUTPUT FORMAT

For each of the 11 tasks, provide:

1. **Key findings** (2-5 most important results with specific paper citations including year and authors)
2. **Direct implications for our work** (what to change, add, or drop)
3. **Priority level** (Critical / High / Medium / Low)
4. **Action items** (specific things to do next)

End with a **synthesis section** that identifies the 3 most important things we're missing and the single most dangerous flaw in our research program.

---

## CONSTRAINTS

- Cite specific papers with authors, year, title, and venue when possible. I need real, verifiable references.
- If you're uncertain about a claim, flag it explicitly rather than hallucinating.
- Prioritize depth over breadth — I'd rather have 5 deeply researched connections than 50 surface-level ones.
- Be adversarial. Assume I have confirmation bias and actively try to break my thesis.
- Think across disciplines: pure math, statistical physics, neuroscience, cognitive science, information theory, algebraic geometry, category theory — anything that genuinely connects.
