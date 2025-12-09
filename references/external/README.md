# External Mechanistic Interpretability References

This folder contains reference materials from outside the mech-interp-latent-lab-phase1 project.

## Files

### `logit_lens_gpt2_nostalgebraist_2020.ipynb`
- **Source:** nostalgebraist (LessWrong blog post, 2020)
- **Original:** https://colab.research.google.com/drive/1-nOE-Qyia3ElM17qrdoHAtGmLCPUZijg
- **Blog post:** https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- **Description:** Classic "logit lens" technique - projecting intermediate activations back into token space to see what the model is "thinking" at each layer
- **Key insight:** GPT-2 activations quickly stop looking like input tokens and converge toward output predictions, refining smoothly across layers
- **Relevance to our work:** Complementary to R_V analysis - shows semantic (token predictions) vs geometric (dimensionality) measures

## Purpose

These are external references that inform our research but aren't part of our codebase. Keeping them separate helps distinguish:
- Our original work (in mech-interp-latent-lab-phase1)
- External references and techniques we're learning from

