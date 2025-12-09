# IOI Circuit – Wang et al., ICLR 2023

- **Title**: Interpretability in the Wild: Identifying Indirect Object Circuits in GPT-2
- **Why relevant**: Gold-standard example of full circuit discovery in a real LM; sets expectations for faithfulness, completeness, and minimality. You can mirror their methodology for a "recursive mode" circuit.
- **Key methods to adapt**:
  - Path patching on specific `(layer, head)` edges.
  - Mean-ablation vs zero-ablation baselines.
  - Large-n dataset (100k+) and bootstrap CIs for effect sizes.
- **Notes**: Use as primary template for how to present a complete, causal story for a circuit.
