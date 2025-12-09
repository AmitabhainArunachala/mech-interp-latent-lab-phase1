# ROME – Meng et al., NeurIPS 2022

- **Title**: Locating and Editing Factual Associations in GPT
- **Why relevant**: Introduces causal tracing (clean→corrupt→restore) and rank-1 editing; useful template for recursive-mode tracing.
- **Key methods to adapt**:
  - Subject-token localisation.
  - Layer-wise causal tracing heatmaps.
- **Notes**: Treat recursive prompts as "facts" about self and trace where they are stored.
