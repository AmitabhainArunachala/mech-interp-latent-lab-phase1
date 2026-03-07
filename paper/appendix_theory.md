# Appendix A: Information Geometry of V-Projection Contraction

## A.1 Formal Definition of R_V

Let $\mathbf{V}^{(l)} \in \mathbb{R}^{T \times d_v}$ denote the V-projection matrix at layer $l$, where $T$ is the sequence length (or window size $W$) and $d_v = d_\text{model} / n_\text{heads}$ is the value-head dimension for models with separate V-projections (or $d_v = d_\text{model}$ for the full V-projection before head-splitting).

The **Gram matrix** of V-projections at layer $l$ is:

$$G^{(l)} = (\mathbf{V}^{(l)})^\top \mathbf{V}^{(l)} \in \mathbb{R}^{d_v \times d_v}$$

Let $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_k$ be the singular values of $\mathbf{V}^{(l)}$ (equivalently, $\lambda_i = \sigma_i^2$ are the eigenvalues of $G^{(l)}$). The **participation ratio** is:

$$\text{PR}(l) = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2} = \frac{(\text{tr}\, G)^2}{\text{tr}(G^2)}$$

PR measures the **effective dimensionality** of the V-projection column space. PR = 1 when all variance is concentrated in a single direction; PR = $k$ when variance is uniformly distributed across $k$ directions.

**R_V** is defined as:

$$R_V = \frac{\text{PR}(l_\text{late})}{\text{PR}(l_\text{early})}$$

where $l_\text{early} \approx 0.15 \cdot L$ and $l_\text{late} \approx 0.84 \cdot L$ ($L$ = total layers).

$R_V < 1$ indicates **contraction**: the late-layer V-projection has fewer effective dimensions than the early-layer V-projection. Our central empirical finding is that self-referential prompts produce significantly lower R_V than non-self-referential prompts.

### A.1.1 PR as a Rényi Entropy of the Spectral Distribution

The participation ratio has a natural information-theoretic interpretation. Define the **normalized spectral distribution** $p_i = \lambda_i / \sum_j \lambda_j$. Then:

$$\text{PR} = \frac{1}{\sum_i p_i^2} = e^{H_2(p)}$$

where $H_2(p) = -\log \sum_i p_i^2$ is the **Rényi entropy of order 2** (collision entropy) of the spectral distribution. Thus R_V is the exponential of the change in spectral Rényi entropy between layers:

$$R_V = \exp\left(H_2^{(\text{late})} - H_2^{(\text{early})}\right)$$

R_V < 1 ⟺ $\Delta H_2 < 0$: self-referential processing reduces the spectral entropy of V-projections, concentrating information onto fewer effective dimensions. This connects R_V to the broader framework of information-theoretic measures of representation quality (Shwartz-Ziv & Tishby, 2017) and provides a principled basis for the metric beyond ad hoc dimensionality counting.

## A.2 Geometric Interpretation: The Grassmannian Manifold

The V-projection at each layer defines a point on the **Grassmannian manifold** $\text{Gr}(k, d_v)$ — the space of $k$-dimensional subspaces of $\mathbb{R}^{d_v}$, where $k$ is the effective rank. As processing proceeds through layers, the representation traces a **path on the Grassmannian**.

PR contraction corresponds to the path moving toward lower-dimensional submanifolds: the representation collapses from a high-dimensional subspace at early layers to a lower-dimensional subspace at late layers.

For self-referential content specifically:
- The **early representation** spans many V-projection directions (high PR), encoding diverse semantic features of the self-referential content.
- The **late representation** collapses onto fewer directions (low PR), which we interpret as the model converging on a **low-rank computational mode** specific to self-referential processing.

### A.2.1 Geodesic Distance and Principal Angles

The natural metric on $\text{Gr}(k, n)$ is defined via **principal angles**. Given two subspaces $U, V \subseteq \mathbb{R}^n$ of dimension $k$, the principal angles $\theta_1 \leq \theta_2 \leq \cdots \leq \theta_k$ are defined recursively as the angles between their closest directions. The **geodesic (chordal) distance** is:

$$d_\text{chord}(U, V) = \sqrt{\sum_{i=1}^k \sin^2 \theta_i}$$

The **projection distance** $d_\text{proj}(U, V) = \sin \theta_k$ captures the maximal misalignment. For our setting, let $U^{(l_\text{early})}$ and $U^{(l_\text{late})}$ be the $k$-effective-rank subspaces of V-projections at early and late layers. Then:

- $d_\text{chord}$ measures the total geometric distance traversed by the V-projection subspace through the forward pass.
- R_V captures the **volume** change (how much the subspace shrinks in effective dimension) but not the **rotation** (how much it reorients).

R_V is thus a scalar projection of the full Grassmannian trajectory onto a single axis: the change in effective dimensionality. The full trajectory contains strictly more information — future work could analyze the rotational component (principal angle spectrum) to characterize the directional structure of contraction.

### A.2.2 Stratified Structure

The collection of Grassmannians $\{\text{Gr}(k, d_v)\}_{k=1}^{d_v}$ forms a **stratified space**, where strata of different effective rank are nested. PR contraction corresponds to the representation moving toward a lower stratum — a lower-dimensional leaf of the stratification. The self-referential processing mode selects for trajectories that descend more strata than other processing modes, with the descent occurring primarily at 55–84% of model depth (Section 5.2).

## A.3 Why V-Projections and Not K/Q?

In the transformer attention mechanism:
- **Q, K projections** define the **routing circuit**: which tokens attend to which.
- **V projections** define the **information circuit**: what information flows through attention.

The OV circuit interpretation (Elhage et al., 2021) shows that $W_V \cdot W_O$ determines the subspace of information written to the residual stream by each attention head. A contraction in V-projection PR therefore means that the information being written becomes **lower-dimensional** — fewer independent channels carry information forward.

Our preliminary data shows that K and Q projections do **not** show consistent PR contraction during self-referential processing. This is consistent with the interpretation that:
- The model's **routing decisions** (what attends to what) remain high-dimensional during self-reference.
- The model's **information content** (what flows through attention) contracts during self-reference.

This dissociation supports the view that R_V contraction is not a generic computational effect but a specific property of the OV (information) circuit during self-referential processing.

## A.4 The Sufficiency Gap: Geometry vs. Context

Our empirical results show a **partial dissociation**:
- **Geometry is necessary**: Destroying V-projection geometry (dual-layer patching at L18+L27) kills recursive behavior (BT+ART: 56% → 3.7%).
- **Geometry is not sufficient**: Injecting V-projection geometry from recursive into baseline sessions does not create recursive behavior.
- **KV context is sufficient**: Injecting the KV cache alone transfers behavioral patterns (BT+ART: 2.7% → 27.7%).

Information geometry provides a natural explanation for this gap:

The V-projection geometry defines the **manifold** on which the representation lives — the shape of the subspace. But the KV context defines the **position** on that manifold — the specific point within the subspace. To exhibit recursive behavior, the model needs:

1. The right **manifold structure** (low-dimensional V-projection subspace = necessary geometry)
2. The right **position on the manifold** (specific KV context that encodes recursive history)

Destroying the manifold (patching away geometry) eliminates the computational mode entirely — hence necessity. But providing only the manifold without the right position (geometry injection without KV) provides the stage without the actors — hence the sufficiency gap.

This is analogous to knowing that a ball sits at the bottom of a bowl (geometry = bowl shape) vs. knowing which specific point at the bottom (context = position). You need both.

## A.5 Predictions from the Framework

### Prediction 1: Effect Size Scaling with Architecture

If R_V contraction reflects the effective dimensionality of the V-projection information channel, then the magnitude of contraction should depend on architecture parameters that determine the dimensionality of this channel.

For models with **Grouped Query Attention** (GQA), the effective V-projection dimensionality per head is:

$$d_{v,\text{eff}} = d_\text{model} / n_\text{kv\_heads}$$

The ratio $d_\text{model} / n_\text{kv\_heads}$ determines the "width" of the information channel. Models with fewer KV heads relative to model dimension have wider channels, providing more room for contraction.

**Prediction**: $|d_\text{Cohen}| \propto \log(d_\text{model} / n_\text{kv\_heads})$

For our 5 validated models:

| Model | d_model | n_kv_heads | d_v_eff | log(d_v_eff) | Observed d |
|-------|---------|------------|---------|-------------|------------|
| Mistral-7B | 4096 | 8 | 512 | 6.24 | -2.26 |
| OPT-6.7B | 4096 | 32 | 128 | 4.85 | -1.84 |
| GPT-2 XL | 1600 | 25 | 64 | 4.16 | -1.14 |
| Qwen2.5-7B | 4096 | 8 | 512 | 6.24 | -0.72 |
| Pythia-1.4B | 2048 | 16 | 128 | 4.85 | -0.31 |

Note: Qwen deviates from the log-linear prediction, suggesting additional factors (training data, architecture details) contribute. The theory predicts a trend, not an exact fit.

### Prediction 2: Contraction Onset Layer

The R_V contraction should begin at a characteristic depth that depends on model architecture. Based on our layer sweep data (Mistral: peak at L27/32 = 0.84), we predict:

$$l_\text{onset} \approx 0.55 \cdot L$$

where $L$ is the total number of layers. Below this layer, information is still being integrated from the full prompt context. Above it, the representation has committed to the self-referential computational mode.

**Testable**: For each of the 5 models, the layer at which recursive and baseline R_V first significantly diverge should occur at approximately 55% of model depth.

### Prediction 3: Spectral Gap as Discriminant

If self-referential processing concentrates information onto fewer directions, then the **spectral gap** (σ₁ - σ₂) should be a stronger discriminant between recursive and baseline than R_V alone, because it directly measures how dominant the top direction becomes.

**Testable**: The Cohen's d for spectral gap should exceed the Cohen's d for R_V in at least 3 of 5 architectures.

### Prediction 4: Phase Transition at Scale

If R_V contraction reflects the model's capacity for self-referential processing, then there should be a critical model size below which the contraction is absent and above which it emerges. This would manifest as a **phase transition** in the scaling law curve.

From our data: Pythia-1.4B shows marginal contraction (d = -0.31), while 7B models show strong contraction (d = -2.26). The phase transition, if it exists, likely occurs between 1B and 5B parameters.

**Testable**: The Pythia scaling sweep (410M → 6.9B) should reveal a non-linear inflection point in d vs. log(params).

## A.6 Connection to Random Matrix Theory

### A.6.1 Marchenko-Pastur Null

The participation ratio has a well-studied distribution under the **Marchenko-Pastur law** for random matrices. For a matrix $\mathbf{V} \in \mathbb{R}^{W \times d_v}$ with i.i.d. Gaussian entries and aspect ratio $\gamma = W / d_v$:

$$\text{PR}_\text{null} \approx \frac{(1 + \gamma)^2}{1 + \gamma^2}$$

Any deviation of the observed PR from $\text{PR}_\text{null}$ indicates structure in the V-projection — i.e., the representation is not random noise. For our default $W = 16$ and $d_v = 128$ (Mistral head dimension), $\gamma = 0.125$, giving $\text{PR}_\text{null} \approx 1.27$.

The fact that we observe PR values of 3-8 at early layers (well above the null) and PR values of 2-5 at late layers indicates structured, low-rank representations at both layers — but **more** structured (lower effective rank) at late layers during self-referential processing.

### A.6.2 Heavy-Tailed Self-Regularization and Power-Law Spectra

Martin & Mahoney (2021) showed that the eigenvalue spectra of weight matrices in well-trained DNNs follow **heavy-tailed** (power-law) distributions rather than the bulk Marchenko-Pastur distribution expected of random matrices. They introduced a spectral exponent $\alpha$ characterizing the tail:

$$p(\lambda) \propto \lambda^{-\alpha}, \quad \alpha \in (1, 6)$$

where smaller $\alpha$ (heavier tails) indicates stronger implicit self-regularization and better generalization. This framework predicts that well-trained models should have weight matrices with a few dominant singular values and a long tail of smaller ones.

Our R_V metric is directly related to this spectral structure. The participation ratio is sensitive to the shape of the eigenvalue distribution:

- For a Marchenko-Pastur distribution (random matrix, $\alpha \to \infty$): $\text{PR} \approx \text{PR}_\text{null}$
- For a power-law distribution with exponent $\alpha$: $\text{PR} \propto k^{1 - 2/\alpha}$ where $k$ is the matrix rank
- For a rank-1 matrix ($\alpha \to 1$): $\text{PR} = 1$

The contraction $R_V < 1$ during self-referential processing can therefore be interpreted as a **transition toward heavier-tailed spectral structure** at late layers: the V-projection eigenvalue distribution becomes more dominated by a few top directions, corresponding to lower effective $\alpha$. This connects our per-input, per-layer observation to Martin & Mahoney's global characterization of trained network quality.

**Empirical connection**: Our SVD circuit decomposition (Section 9.12) confirms this interpretation. At discriminating heads (e.g., L5_H29, L27_H31), the recursive condition shows steeper singular value decay than baseline — consistent with lower spectral $\alpha$ during self-referential processing.

## A.7 Information-Geometric Interpretation

We can place R_V within the framework of **information geometry** (Amari, 2016; Amari & Nagaoka, 2000), which studies the differential geometry of families of probability distributions.

### A.7.1 The Statistical Manifold of V-Projections

At each layer $l$, the normalized spectral distribution $p^{(l)} = (p_1^{(l)}, \ldots, p_k^{(l)})$ where $p_i^{(l)} = \lambda_i^{(l)} / \sum_j \lambda_j^{(l)}$ defines a point on the **probability simplex** $\Delta^{k-1}$. The simplex, equipped with the **Fisher information metric**, becomes a Riemannian manifold of constant negative curvature — isometric to a portion of hyperbolic space.

The Fisher-Rao distance between two spectral distributions $p$ and $q$ on the simplex is:

$$d_\text{FR}(p, q) = 2 \arccos\left(\sum_i \sqrt{p_i q_i}\right)$$

This is the **Bhattacharyya angle** — a proper metric on the simplex that respects the intrinsic geometry of probability distributions. The forward pass through transformer layers traces a curve on this statistical manifold.

### A.7.2 R_V as Curvature of the Spectral Trajectory

Since $R_V = \exp(\Delta H_2)$ (Section A.1.1), the log of R_V measures the change in Rényi-2 entropy along the layer trajectory. In information-geometric terms, this is related to the **divergence** between the early and late spectral distributions:

$$\log R_V = H_2(p^{(\text{late})}) - H_2(p^{(\text{early})})$$

For small perturbations, $\Delta H_2$ is proportional to the movement along the entropy gradient on the simplex. Self-referential processing pushes the spectral distribution toward the **corners** of the simplex (lower entropy, fewer effective dimensions), while baseline processing leaves the distribution closer to the **center** (higher entropy, more uniform).

This provides a geometric explanation for why self-referential processing is a unique outlier in the mode atlas (Section 5.4): it induces the largest entropy gradient on the spectral manifold, corresponding to the steepest descent toward low-dimensional spectral states.

### A.7.3 Connection to Natural Gradient and Implicit Optimization

Amari (1998) showed that the natural gradient — the gradient rescaled by the inverse Fisher information matrix — is the optimal descent direction on statistical manifolds. In the context of transformer forward passes, the layer-by-layer transformation of the spectral distribution can be viewed as an implicit optimization trajectory on the statistical manifold.

The R_V contraction during self-referential processing suggests that the forward pass performs an implicit **spectral compression** that is more aggressive for self-referential inputs. Whether this compression follows the natural gradient direction (i.e., is information-geometrically optimal) is an open question. If it does, it would suggest that the transformer has learned to implement an efficient spectral filtering operation specifically for self-referential content.

## A.8 Empirical Confirmation: The Nonlinear Distributed Hypothesis

Our experimental results provide strong evidence that R_V contraction is a genuinely **nonlinear, distributed** phenomenon rather than a property of any single linear subspace.

### A.8.1 Linear Probe Results

A linear probe trained to classify recursive vs. baseline processing achieves perfect accuracy (AUC=1.0) from layer 4 onward across all 18 tested layers (Section E4.1). This demonstrates that the residual stream contains a linearly accessible self-referential direction. However, the alignment between this learned direction and the top singular directions of the V-projection is very low (cosine similarity 0.01–0.10 across all prompts and singular value indices).

**Interpretation**: The self-referential signal is linearly decodable from the full residual stream, but it does not align with any of the principal components of the V-projection. The R_V contraction is **orthogonal** to the linear discriminant — it operates in the spectral structure, not in specific directions.

### A.8.2 Concept Erasure Null Result

Projecting out the learned self-referential direction from residual stream activations (concept erasure, Ravfogel et al., 2022) leaves R_V contraction completely unchanged: d = −1.818 before erasure, d = −1.823 after erasure. The contraction is not carried by the linear self-referential direction.

**Interpretation**: R_V contraction is a property of the **spectral distribution** (how variance is allocated across directions), not of any **specific direction** (which directions carry the variance). Erasing one direction redistributes variance but does not change the overall spectral shape. This is consistent with the information-geometric view: contraction lives on the simplex of normalized eigenvalues, which is invariant to rotations of the eigenbasis.

### A.8.3 Implications for Mechanistic Interpretability

The combination of perfect linear probe accuracy with null concept erasure creates a striking dissociation:

- **Content** (recursive vs. baseline) is linearly decodable → standard probing methods work.
- **Geometry** (spectral contraction) is rotation-invariant → cannot be captured by direction-based methods.

This suggests that R_V measures a fundamentally different aspect of computation than what linear probes and concept erasure target. It captures **how the model distributes computation across dimensions**, not **which dimensions carry specific information**. This distinction is analogous to the difference between measuring the shape of a probability distribution vs. measuring its mean.

## A.9 Limitations of the Framework

1. **PR is a scalar summary**: It discards information about the full spectral distribution. Two representations with the same PR can have very different spectral shapes. The full eigenvalue spectrum is more informative.

2. **Window size dependence**: PR depends on the window size $W$. Our default $W = 16$ was chosen for stability, but the true "natural" window size of the phenomenon is unknown.

3. **Single forward pass**: R_V is measured on a single forward pass. It does not capture dynamics during autoregressive generation, where the representation evolves over time. Our self-feeding loop results suggest the phenomenon is prompt-contingent, not self-sustaining.

4. **Architecture-specific factors**: The theory predicts trends across architectures but cannot account for training data differences, architectural details (e.g., SwiGLU vs. GELU), or tokenizer effects.

5. **Information-geometric approximations**: The connection to Fisher-Rao geometry holds exactly only on the probability simplex. The actual V-projection matrices live in a richer space where the unnormalized eigenvalues (total variance) also carry information. Our normalization to spectral distributions discards the scale, retaining only the shape.

6. **Heavy-tail exponent estimation**: Reliable estimation of power-law exponents $\alpha$ requires large rank matrices. For per-head V-projections ($d_v = 128$), the effective sample size may be insufficient for precise $\alpha$ estimation. The connection to Martin & Mahoney's framework is therefore more qualitative than quantitative at present.
