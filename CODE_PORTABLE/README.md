# Logit Lens - nostalgebraist (2020)

This folder contains the logit lens notebook code from nostalgebraist.

## Source

- **Author:** nostalgebraist
- **Original Colab:** https://colab.research.google.com/drive/1-nOE-Qyia3ElM17qrdoHAtGmLCPUZijg
- **Blog Post:** https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

## What It Does

Projects intermediate GPT-2 activations back into token space to see what the model is "thinking" at each layer.

## Key Technique

Multiply hidden states by `wte^T` (transpose of embedding matrix) to convert activations → logits → token predictions.

## Files

- `logit_lens_gpt2_nostalgebraist_2020.ipynb` - Complete notebook with all code

## Usage

Open in Jupyter/Colab and run cells. The notebook is self-contained with all setup code included.
