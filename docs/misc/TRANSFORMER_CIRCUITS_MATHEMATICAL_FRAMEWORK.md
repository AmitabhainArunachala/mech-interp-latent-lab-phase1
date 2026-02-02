# A Mathematical Framework for Transformer Circuits

**Authors:** Nelson Elhage*, Neel Nanda*, Catherine Olsson*, Tom Henighan†, Nicholas Joseph†, Ben Mann†, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, Chris Olah‡

**Affiliation:** Anthropic

**Published:** December 22, 2021

*Core Research Contributor; † Core Infrastructure Contributor; ‡ Correspondence to colah@anthropic.com

---

## Contents

- [Summary of Results](#summary-of-results)
- [Transformer Overview](#transformer-overview)
- [Zero-Layer Transformers](#zero-layer-transformers)
- [One-Layer Attention-Only Transformers](#one-layer-attention-only-transformers)
- [Two-Layer Attention-Only Transformers](#two-layer-attention-only-transformers)
- [Where Does This Leave Us?](#where-does-this-leave-us)
- [Related Work](#related-work)
- [Comments](#comments)
- [Acknowledgments](#acknowledgments)
- [Author Contributions](#author-contributions)
- [Citation Information](#citation-information)
- [Additional Intuition and Observations](#additional-intuition-and-observations)
- [Notation](#notation)
- [Technical Details](#technical-details)

---

## Summary of Results

### Reverse Engineering Results

To explore the challenge of reverse engineering transformers, we reverse engineer several toy, attention-only models. In doing so we find:

1. **Zero layer transformers** model bigram statistics. The bigram table can be accessed directly from the weights.

2. **One layer attention-only transformers** are an ensemble of bigram and "skip-trigram" (sequences of the form "A… B C") models. The bigram and skip-trigram tables can be accessed directly from the weights, without running the model. These skip-trigrams can be surprisingly expressive. This includes implementing a kind of very simple in-context learning.

3. **Two layer attention-only transformers** can implement much more complex algorithms using compositions of attention heads. These compositional algorithms can also be detected directly from the weights. Notably, two layer models use attention head composition to create "induction heads", a very general in-context learning algorithm.

4. **One layer and two layer attention-only transformers** use very different algorithms to perform in-context learning. Two layer attention heads use qualitatively more sophisticated inference-time algorithms — in particular, a special type of attention head we call an induction head — to perform in-context-learning, forming an important transition point that will be relevant for larger models.

### Conceptual Take-Aways

We've found that many subtle details of the transformer architecture require us to approach reverse engineering it in a pretty different way from how the InceptionV1 Circuits work. We'll unpack each of these points in the sections below, but for now we briefly summarize:

1. **Attention heads can be understood as independent operations**, each outputting a result which is added into the residual stream. Attention heads are often described in an alternate "concatenate and multiply" formulation for computational efficiency, but this is mathematically equivalent.

2. **Attention-only models can be written as a sum of interpretable end-to-end functions** mapping tokens to changes in logits. These functions correspond to "paths" through the model, and are linear if one freezes the attention patterns.

3. **Transformers have an enormous amount of linear structure.** One can learn a lot simply by breaking apart sums and multiplying together chains of matrices.

4. **Attention heads can be understood as having two largely independent computations:** a QK ("query-key") circuit which computes the attention pattern, and an OV ("output-value") circuit which computes how each token affects the output if attended to.

5. **Key, query, and value vectors can be thought of as intermediate results** in the computation of the low-rank matrices W_Q^T W_K and W_O W_V. It can be useful to describe transformers without reference to them.

6. **Composition of attention heads greatly increases the expressivity of transformers.** There are three different ways attention heads can compose, corresponding to keys, queries, and values. Key and query composition are very different from value composition.

7. **All components of a transformer** (the token embedding, attention heads, MLP layers, and unembedding) communicate with each other by reading and writing to different subspaces of the residual stream. Rather than analyze the residual stream vectors, it can be helpful to decompose the residual stream into all these different communication channels, corresponding to paths through the model.

---

## Transformer Overview

Before we attempt to reverse engineer transformers, it's helpful to briefly review the high-level structure of transformers and describe how we think about them.

In many cases, we've found it helpful to reframe transformers in equivalent, but non-standard ways. Mechanistic interpretability requires us to break models down into human-interpretable pieces. An important first step is finding the representation which makes it easiest to reason about the model.

### Model Simplifications

To demonstrate the ideas in this paper in their cleanest form, we focus on "toy transformers" with some simplifications:

- **Attention-only transformers:** We focus on transformers without MLP layers. This is a very dramatic simplification, but allows us to give an especially elegant treatment of attention head circuits.

- **No biases:** A model with biases can always be simulated without them by folding them into the weights.

- **No layer normalization:** Up to a variable scaling, layer norm can be merged into adjacent weights.

### High-Level Architecture

A transformer starts with a token embedding, followed by a series of "residual blocks", and finally a token unembedding. Each residual block consists of an attention layer, followed by an MLP layer. Both the attention and MLP layers each "read" their input from the residual stream (by performing a linear projection), and then "write" their result to the residual stream by adding a linear projection back in. Each attention layer consists of multiple heads, which operate in parallel.

### Virtual Weights and the Residual Stream as a Communication Channel

The residual stream is simply the sum of the output of all the previous layers and the original embedding. We generally think of the residual stream as a communication channel, since it doesn't do any processing itself and all layers communicate through it.

The residual stream has a deeply linear structure. Every layer performs an arbitrary linear transformation to "read in" information from the residual stream at the start, and performs another arbitrary linear transformation before adding to "write" its output back into the residual stream.

**Virtual Weights:** One can think of implicit "virtual weights" directly connecting any pair of layers (even those separated by many other layers), by multiplying out their interactions through the residual stream. These virtual weights are the product of the output weights of one layer with the input weights of another (i.e., W_I2 W_O1), and describe the extent to which a later layer reads in the information written by a previous layer.

### Attention Heads are Independent and Additive

We think of transformer attention layers as several completely independent attention heads h ∈ H which operate completely in parallel and each add their output back into the residual stream. This is mathematically equivalent to the standard "concatenate and multiply" formulation, but more interpretable.

### Attention Heads as Information Movement

The fundamental action of attention heads is moving information. They read information from the residual stream of one token, and write it to the residual stream of another token. The main observation is that which tokens to move information from is completely separable from what information is "read" to be moved and how it is "written" to the destination.

Using tensor notation, we can describe attention as:

```
h(x) = (A ⊗ W_O W_V) · x
```

Where:
- A mixes across tokens (attention pattern)
- W_O W_V acts on each vector independently (OV circuit)

The attention pattern A = softmax(x^T W_Q^T W_K x) can be computed without referring to keys and queries separately.

**Key Observations:**
- Attention heads move information from the residual stream of one token to another
- An attention head applies two linear operations, A and W_O W_V, which operate on different dimensions and act independently
- A governs which token's information is moved from and to
- W_O W_V governs which information is read from the source token and how it is written to the destination token
- W_Q^T W_K and W_O W_V can always be thought of as individual, low-rank matrices

---

## Zero-Layer Transformers

A "zero-layer" transformer takes a token, embeds it, unembeds it to produce logits predicting the next token:

```
T = W_U W_E
```

Because the model cannot move information from other tokens, we are simply predicting the next token from the present token. This means that the optimal behavior of W_U W_E is to approximate the bigram log-likelihood.

---

## One-Layer Attention-Only Transformers

We claim that one-layer attention-only transformers can be understood as an ensemble of a bigram model and several "skip-trigram" models (affecting the probabilities of sequences "A… BC"). Intuitively, this is because each attention head can selectively attend from the present token ("B") to a previous token ("A") and copy information to adjust the probability of possible next tokens ("C").

### The Path Expansion Trick

Our key trick is to expand the product of layers into a sum where every term corresponds to an end-to-end path:

```
T(x) = (Id ⊗ W_U W_E) · x + Σ_h (A_h ⊗ (W_U W_OV^h W_E)) · x
```

Each of these end-to-end path terms is tractable to understand, can be reasoned about independently, and additively combine to create model behavior.

### Splitting Attention Head terms into Query-Key and Output-Value Circuits

For each attention head h we have a term A_h ⊗ (W_U W_OV^h W_E) where A_h = softmax(t^T · W_E^T W_QK^h W_E · t).

These terms consist of two separable operations:

1. **W_E^T W_QK^h W_E** — The "query-key (QK) circuit." It provides the attention score for every query and key token.

2. **W_U W_OV^h W_E** — The "Output-Value (OV) circuit." It describes how a given token will affect the output logits if attended to.

### Interpretation as Skip-Trigrams

Together, the three tokens involved form a "skip-trigram" of the form [source]... [destination][out], and the "out" is modified.

### Copying / Primitive In-Context Learning

Most attention heads in one layer models dedicate an enormous fraction of their capacity to copying. The OV circuit sets things up so that tokens, if attended to by the head, increase the probability of that token, and to a lesser extent, similar tokens. The QK circuit then only attends back to tokens which could plausibly be the next token.

### Other Interesting Skip-Trigrams

Skip-trigrams can produce more complex behavior than one might expect. Examples include:
- Python indentation patterns
- Function argument patterns
- Library-specific patterns
- Common phrases and constructions
- URL schemes

### Do We "Fully Understand" One-Layer Models?

We now understand this simplified model in the same sense that one might look at the weights of a giant linear regression and understand it. There's no longer any algorithmic mystery. The contextualization problem of neural network parameters has been stripped away. But without further work on summarizing it, there's far too much there for one to hold the model in their head.

---

## Two-Layer Attention-Only Transformers

Composition of attention heads is the key difference between one-layer and two-layer attention-only transformers. Without composition, a two-layer model would simply have more attention heads to implement skip-trigrams with. But in practice, two-layer models discover ways to exploit attention head composition to express a much more powerful mechanism for accomplishing in-context learning.

### Three Kinds of Composition

When attention heads compose, there are three options:

1. **Q-Composition:** W_Q reads in a subspace affected by a previous head.
2. **K-Composition:** W_K reads in a subspace affected by a previous head.
3. **V-Composition:** W_V reads in a subspace affected by a previous head.

Q- and K-Composition are quite different from V-Composition. Q- and K-Composition both affect the attention pattern, allowing attention heads to express much more complex patterns. V-Composition, on the other hand, affects what information an attention head moves when it attends to a given position; the result is that V-composed heads really act more like a single unit and can be thought of as creating an additional "virtual attention heads".

### Path Expansion of Logits

Following our approach to the one-layer model, we write out a product where every term is a layer in the model, and expand to create a sum where every term is an end-to-end path through the model.

### Path Expansion of Attention Scores QK Circuit

For second layer QK-circuits, both Q-composition and K-composition come into play, with previous layer attention heads potentially influencing the construction of the keys and queries.

### Induction Heads

In small two-layer attention-only transformers, composition seems to be primarily used for one purpose: the creation of what we call **induction heads**.

**Function of Induction Heads:** Induction heads search over the context for previous examples of the present token. If they don't find it, they attend to the first token and do nothing. But if they do find it, they then look at the next token and copy it. This allows them to repeat previous sequences of tokens, both exactly and approximately.

**How Induction Heads Work:** The central trick is that the key is computed from tokens shifted one token back. The query searches for "similar" key vectors, but because keys are shifted, finds the next token.

The minimal way to create an induction head is to use K-composition with a previous token head to shift the key vector forward one token. This creates a term of the form Id ⊗ A_{h-1} ⊗ W in the QK-circuit.

**Mechanistic Theory:** Induction heads must:
1. Have a "copying" OV circuit matrix.
2. Have a "same matching" QK circuit matrix associated with the Id ⊗ A_{h-1} ⊗ W term.

### Virtual Attention Heads

Virtual attention heads are the terms corresponding to the V-composition of two heads. They have their own attention pattern A_{h2∘h1} = A_{h2} A_{h1} and their own OV matrix W_{OV}^{h2∘h1} = W_{OV}^{h2} W_{OV}^{h1}.

This kind of composition seems quite powerful, and there are a lot of virtual attention heads. The number of normal heads grows linearly in the number of layers, while the number of virtual heads based on the composition of two heads grows quadratically, on three heads grows cubically, etc.

---

## Where Does This Leave Us?

Over the last few sections, we've made progress on understanding one-layer and two-layer attention-only transformers. But our ultimate goal is to understand transformers in general.

**Relevance to Larger Models:** Normal transformers contain some circuits which appear to be primarily attentional. Even in the presence of MLP layers, attention heads still operate on the residual stream and can still interact directly with each other and with the embeddings. In practice, we find instances of interpretable circuits involving only attention heads and the embeddings.

We actually see some analogous attention heads and circuits in large models to those we analyzed in these toy models! In particular, large models form many induction heads, and that the basic building block of their construction is K-composition with a previous token head, just as we saw here. This appears to be a central driver of in-context learning in language models of all sizes.

**Limitations:** We can probably only understand a small portion of large language models this way. MLP layers make up 2/3rds of a standard transformer's parameters. More complete understanding will require progress on MLP layers.

---

## Related Work

### Circuits

The Distill Circuits thread was a concerted effort to reverse engineer the InceptionV1 model. Our work seeks to do something similar for large language models.

### The Logit Lens

Previous work by the LessWrong user Nostalgebraist on a method they call the "Logit Lens" explores the same linear structure of the residual stream we heavily exploit.

### Attention Head Analysis

Our work follows the lead of several previous papers in exploring investigating transformer attention heads. The largest difference between these previous analyses and our work really seems to be a matter of goals: we seek to provide an end-to-end mechanistic account, rather than empirically describe attention patterns.

### Criticism of Attention as Explanation

Our framework might be thought of as offering — for the limited case of attention-only models — a typology of ways in which naive interpretation of attention patterns can be misleading, and a specific way in which they can be correct.

---

## Comments

### Summary of Follow-Up Research

Since the publication of this paper, a significant amount of follow-up work has greatly clarified and extended the preliminary ideas:

- **Understanding MLP Layers and Superposition:** More has been learned about MLP layer neurons, there has been significant elaboration on the theory of superposition, and alternative theories competing with superposition have been proposed.

- **Attention Head Composition and Circuits:** A preliminary investigation explored the idea of attention head composition in more detail. A paper described a complex circuit of attention heads.

- **Induction Heads:** A follow-up paper explored how much induction heads contribute to in-context learning. A number of researchers reproduced the general results about induction heads.

### Correction: Attention Head Composition Diagram

Following the publication of this paper, we became aware of a bug in an underlying library we wrote. This only affected one diagram, but does impact on our interpretation of the "Two-Layer Attention Only Transformers" section in some ways. In particular, there is more attention head composition going on in that model than it seemed.

---

## Acknowledgments

In writing this paper, our thinking was greatly clarified and encouraged by correspondence with Martin Wattenberg, Vladimir Mikulik, Jeff Wu, Evan Hubinger, and Peter Hase.

---

## Author Contributions

**Theoretical Framework:** Developed as part of ongoing conversations and empirical investigation between Nelson Elhage, Catherine Olsson, Neel Nanda, and Chris Olah.

**Analysis of Toy Models:** Done by Chris Olah and Neel Nanda, heavily based on previous experiments by Nelson Elhage and Catherine Olsson.

**Writing:** Drafted by Chris Olah. Neel Nanda made major pedagogical improvements. Dario Amodei contributed heavily to the high-level framing.

---

## Citation Information

**Please cite as:**

Elhage, et al., "A Mathematical Framework for Transformer Circuits", Transformer Circuits Thread, 2021.

**BibTeX:**

```bibtex
@article{elhage2021mathematical,
   title={A Mathematical Framework for Transformer Circuits},
   author={Elhage, Nelson and Nanda, Neel and Olsson, Catherine and Henighan, Tom and Joseph, Nicholas and Mann, Ben and Askell, Amanda and Bai, Yuntao and Chen, Anna and Conerly, Tom and DasSarma, Nova and Drain, Dawn and Ganguli, Deep and Hatfield-Dodds, Zac and Hernandez, Danny and Jones, Andy and Kernion, Jackson and Lovitt, Liane and Ndousse, Kamal and Amodei, Dario and Brown, Tom and Clark, Jack and Kaplan, Jared and McCandlish, Sam and Olah, Chris},
   year={2021},
   journal={Transformer Circuits Thread},
   note={https://transformer-circuits.pub/2021/framework/index.html}
}
```

---

## Additional Intuition and Observations

### MLP Layers

This article has focused on attention-only transformers, without MLP layers. In theory, there's a lot of reason to be optimistic about understanding MLP neurons. They have an activation function which should encourage features to align with the basis dimensions. They're four times larger than the residual stream, and information doesn't need to flow through them, which are both factors one might expect to reduce polysemanticity. Unfortunately, things are much more challenging in practice.

### Virtual Weights and Convolution-like Structure

When we apply path expansion to various terms in a transformer, we generally get virtual weights that can be seen as a generalization of a convolution, with attention heads taking the place of relative position.

### Activation Properties

We often find it helpful to think differently about various activations in transformers based on whether they have:

- **Privileged Basis vs Basis Free:** A privileged basis occurs when some aspect of a model's architecture encourages neural network features to align with basis dimensions.

- **Bottleneck Activations:** An activation is a bottleneck activation if it is a lower-dimensional intermediate between two higher dimensional activations.

---

## Notation

### Variable Definitions

**Main Model Activations and Parameters:**
- T(t): Transformer logits
- t: One-hot encoded tokens
- x_n: Residual stream vectors at layer n
- W_E: Token embedding
- W_U: Unembedding / softmax weights

**Attention Heads:**
- H_n: Set of attention heads at layer n
- h(x): Output of attention head h
- A_h: Attention pattern of head h
- W_Q^h, W_K^h, W_V^h: Query, key, value weights
- W_O^h: Output weights
- W_OV^h = W_O^h W_V^h
- W_QK^h = W_Q^T^h W_K^h

**MLP Layers:**
- m(x): Output of MLP layer m
- a_m: Activations of MLP layer m
- W_I^m: Input weights
- W_O^m: Output weights

### Tensor Product / Kronecker Product Notation

We denote tensor products with the ⊗ symbol:

- A product like Id ⊗ W represents multiplying each position in our context by a matrix.
- A product like A ⊗ Id represents multiplying across positions.
- A product like A ⊗ W multiplies the vector at each position by W and across positions with A.

The products obey the mixed-product property: (A ⊗ B) · (C ⊗ D) = (AC) ⊗ (BD).

---

## Technical Details

### Model Details

The models used as examples in this paper are zero, one, and two layer decoder-only, attention-only transformers. For all models, d_model = n_heads * d_head, typically with n_heads = 12 and d_head = 64.

Models have a context size of 2048 tokens and use dense attention. We use a positional mechanism similar to Press et al., adding sinusoidal embeddings immediately before multiplying by W_Q and W_K to produce queries and keys.

### Handling Layer Normalization

Layer normalization adds complexity, but up to a variable scaling, layer normalization applies a fixed affine transformation. We can fold everything but normalization into adjacent parameters, and then think of the normalization scaling as a variable reweighting of the set of path terms going through that layer normalization.

### Working with Low-Rank Matrices

In this paper, we find ourselves dealing with very large, but extremely low-rank matrices. We recommend:

- Keeping matrices in factored form whenever possible
- Exploiting the fact that λ_i(AB) = λ_i(BA) for eigenvalues
- Using efficient SVD algorithms for low-rank matrices

---

**Source:** [Transformer Circuits Thread](https://transformer-circuits.pub/2021/framework/index.html)

