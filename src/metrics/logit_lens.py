"""
Logit Lens: Per-layer token predictions.

At each layer, apply final LayerNorm + LM head to see what the model
"thinks" at that point in processing.

Reference: nostalgebraist (2020), "interpreting GPT: the logit lens"
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class LogitLensResult:
    """Result for a single layer."""
    layer: int
    top_tokens: List[str]  # Top-K tokens
    top_probs: List[float]  # Their probabilities
    top_ids: List[int]  # Token IDs
    entropy: float  # Prediction entropy (lower = more confident)


def compute_logit_lens(
    model,
    tokenizer,
    hidden_states: Tuple[torch.Tensor, ...],
    target_position: int = -1,
    top_k: int = 5,
    device: str = "cuda",
) -> List[LogitLensResult]:
    """
    Compute logit lens for all layers.
    
    Args:
        model: HuggingFace model with model.model.norm and model.lm_head
        tokenizer: Tokenizer for decoding
        hidden_states: Tuple of hidden states from model(..., output_hidden_states=True)
                      Shape per layer: (batch, seq_len, hidden_dim)
        target_position: Which position to analyze (default: -1 = last token)
        top_k: Number of top tokens to return per layer
        device: Device for computation
    
    Returns:
        List of LogitLensResult, one per layer (including embedding layer 0)
    """
    results = []
    
    for layer_idx, h in enumerate(hidden_states):
        # Get hidden state at target position
        # h shape: (batch, seq_len, hidden_dim)
        h_pos = h[0, target_position, :]  # (hidden_dim,)
        
        # Apply final LayerNorm (critical for logit lens!)
        h_norm = model.model.norm(h_pos)
        
        # Project to vocabulary
        logits = model.lm_head(h_norm)  # (vocab_size,)
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Top-K
        top_probs, top_ids = torch.topk(probs, k=top_k)
        
        # Decode tokens
        top_tokens = [tokenizer.decode([tid.item()]) for tid in top_ids]
        
        # Compute entropy (measure of uncertainty)
        # H = -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum().item()
        
        results.append(LogitLensResult(
            layer=layer_idx,
            top_tokens=top_tokens,
            top_probs=top_probs.cpu().tolist(),
            top_ids=top_ids.cpu().tolist(),
            entropy=entropy,
        ))
    
    return results


def compute_logit_lens_trajectory(
    model,
    tokenizer,
    text: str,
    target_position: int = -1,
    top_k: int = 5,
    device: str = "cuda",
) -> Tuple[List[LogitLensResult], Dict[str, Any]]:
    """
    Full logit lens analysis for a text.
    
    Returns:
        results: List of LogitLensResult per layer
        metadata: Dict with prompt info, crystallization layer, etc.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    results = compute_logit_lens(
        model, tokenizer, outputs.hidden_states,
        target_position=target_position, top_k=top_k, device=device
    )
    
    # Find crystallization point (where top token stabilizes)
    final_token = results[-1].top_tokens[0]
    crystallization_layer = None
    for r in results:
        if r.top_tokens[0] == final_token:
            crystallization_layer = r.layer
            break
    
    # Find entropy minimum (highest confidence)
    min_entropy_layer = min(results, key=lambda r: r.entropy).layer
    
    metadata = {
        "prompt": text,
        "target_position": target_position,
        "final_prediction": final_token,
        "final_prob": results[-1].top_probs[0],
        "crystallization_layer": crystallization_layer,
        "min_entropy_layer": min_entropy_layer,
        "min_entropy": min(r.entropy for r in results),
    }
    
    return results, metadata


def find_recursive_emergence(
    results: List[LogitLensResult],
    recursive_tokens: List[str],
) -> Dict[str, Any]:
    """
    Find where recursive tokens first appear in top predictions.
    
    Args:
        results: Logit lens results
        recursive_tokens: List of recursive token strings to look for
                         e.g., ["self", "itself", "observer", "awareness"]
    
    Returns:
        Dict with emergence info per token
    """
    recursive_tokens_lower = [t.lower().strip() for t in recursive_tokens]
    emergence = {}
    
    for token in recursive_tokens_lower:
        first_appearance = None
        max_prob = 0.0
        max_prob_layer = None
        
        for r in results:
            for i, pred_token in enumerate(r.top_tokens):
                if token in pred_token.lower().strip():
                    if first_appearance is None:
                        first_appearance = r.layer
                    if r.top_probs[i] > max_prob:
                        max_prob = r.top_probs[i]
                        max_prob_layer = r.layer
        
        emergence[token] = {
            "first_appearance": first_appearance,
            "max_prob": max_prob,
            "max_prob_layer": max_prob_layer,
        }
    
    return emergence
