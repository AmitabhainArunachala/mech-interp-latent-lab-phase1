"""
Per-token R_V measurement during autoregressive generation.

This module enables tracking geometric contraction (R_V) token-by-token
as the model generates text, allowing us to correlate R_V dynamics with
behavioral markers (L4 tokens, deflection patterns, etc.).

Key insight: If R_V contraction CAUSES recursive behavior, we should see
R_V drop BEFORE L4 tokens appear, not just at prompt encoding.
"""

import logging
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field

import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.hooks import capture_v_projection

logger = logging.getLogger(__name__)


@dataclass
class TokenRVStep:
    """Single step in per-token R_V trajectory."""
    step_idx: int
    token_id: int
    token_str: str
    rv: float
    pr_early: float
    pr_late: float
    seq_len: int  # Total sequence length at this step


@dataclass
class GenerationRVTrajectory:
    """Complete R_V trajectory for a generation."""
    prompt: str
    prompt_rv: float  # R_V measured on prompt before generation
    steps: List[TokenRVStep] = field(default_factory=list)
    generated_text: str = ""
    
    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "prompt_rv": self.prompt_rv,
            "generated_text": self.generated_text,
            "steps": [
                {
                    "step_idx": s.step_idx,
                    "token_id": s.token_id,
                    "token_str": s.token_str,
                    "rv": s.rv,
                    "pr_early": s.pr_early,
                    "pr_late": s.pr_late,
                    "seq_len": s.seq_len,
                }
                for s in self.steps
            ],
            "min_rv": min([s.rv for s in self.steps]) if self.steps else float("nan"),
            "max_rv": max([s.rv for s in self.steps]) if self.steps else float("nan"),
            "mean_rv": np.mean([s.rv for s in self.steps]) if self.steps else float("nan"),
        }


def participation_ratio_from_hidden(
    hidden: torch.Tensor,
    window: int = 16,
) -> float:
    """
    Compute PR from hidden states tensor.
    
    Args:
        hidden: Hidden states tensor (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        window: Window size for PR calculation
        
    Returns:
        PR value (float)
    """
    if hidden is None:
        return float("nan")
    
    # Handle batch dimension
    if hidden.dim() == 3:
        hidden = hidden[0]
    
    T, D = hidden.shape
    
    if T < window:
        # For short sequences, use all available tokens
        W = T
    else:
        W = window
    
    # Extract last W tokens
    h_window = hidden[-W:, :].double()
    
    try:
        # SVD
        U, S, Vt = torch.linalg.svd(h_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        
        # Check for degeneracy
        total_variance = S_sq.sum()
        if total_variance < 1e-10:
            return float("nan")
        
        # PR = (sum^2) / sum(^2)
        pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
        return float(pr)
    except Exception:
        return float("nan")


def compute_rv_from_hidden_states(
    hidden_states: Tuple[torch.Tensor],
    early_layer: int,
    late_layer: int,
    window: int = 16,
) -> Tuple[float, float, float]:
    """
    Compute R_V from hidden states tuple (as returned by model forward with output_hidden_states=True).
    
    Args:
        hidden_states: Tuple of hidden state tensors, one per layer
        early_layer: Early layer index
        late_layer: Late layer index
        window: Window size for PR calculation
        
    Returns:
        (rv, pr_early, pr_late)
    """
    if hidden_states is None or len(hidden_states) <= max(early_layer, late_layer):
        return (float("nan"), float("nan"), float("nan"))
    
    h_early = hidden_states[early_layer]
    h_late = hidden_states[late_layer]
    
    pr_early = participation_ratio_from_hidden(h_early, window)
    pr_late = participation_ratio_from_hidden(h_late, window)
    
    if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
        return (float("nan"), float("nan"), float("nan"))
    
    rv = pr_late / pr_early
    return (rv, pr_early, pr_late)


def generate_with_rv_tracking(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    early_layer: int = 5,
    late_layer: Optional[int] = None,
    window: int = 16,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = "cuda",
    measure_interval: int = 1,  # Measure R_V every N tokens (1 = every token)
) -> GenerationRVTrajectory:
    """
    Generate text with per-token R_V tracking.
    
    This function hooks into the autoregressive loop and measures R_V at each step,
    allowing us to correlate geometric dynamics with behavioral markers in real-time.
    
    Args:
        model: The transformer model (must be in eval mode)
        tokenizer: The tokenizer
        prompt: Input prompt text
        early_layer: Early layer for R_V computation
        late_layer: Late layer for R_V computation (auto-derived if None)
        window: Window size for PR calculation
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to sample or use greedy decoding
        device: Target device
        measure_interval: Measure R_V every N tokens (for performance)
        
    Returns:
        GenerationRVTrajectory containing prompt R_V and per-token R_V trajectory
        
    Note:
        This is computationally expensive (~2x slower than normal generation)
        because we run a forward pass with output_hidden_states=True at each step.
    """
    # Auto-derive late layer
    num_layers = getattr(model.config, "num_hidden_layers", 32)
    if late_layer is None:
        late_layer = num_layers - 5
    
    # Measure R_V on prompt first
    from .rv import compute_rv_with_components
    prompt_rv, _, _ = compute_rv_with_components(
        model, tokenizer, prompt, early_layer, late_layer, window, device
    )
    
    trajectory = GenerationRVTrajectory(
        prompt=prompt,
        prompt_rv=prompt_rv,
    )
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    generated_tokens = []
    past_key_values = None
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass with hidden states
            if past_key_values is None:
                # First step: full prompt
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=True,
                )
            else:
                # Subsequent steps: only last token
                outputs = model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # Last token logits
            
            # Sample next token
            if do_sample:
                # Apply temperature
                logits = logits / temperature
                
                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), dtype=torch.long, device=device)
            ], dim=-1)
            
            generated_tokens.append(next_token.item())
            
            # Measure R_V every measure_interval steps
            if step % measure_interval == 0 or step == max_new_tokens - 1:
                # Compute R_V from hidden states
                rv, pr_e, pr_l = compute_rv_from_hidden_states(
                    outputs.hidden_states,
                    early_layer,
                    late_layer,
                    window,
                )
                
                token_str = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                
                step_record = TokenRVStep(
                    step_idx=step,
                    token_id=next_token.item(),
                    token_str=token_str,
                    rv=rv,
                    pr_early=pr_e,
                    pr_late=pr_l,
                    seq_len=input_ids.shape[1],
                )
                trajectory.steps.append(step_record)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode full generated text
    trajectory.generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return trajectory


def analyze_rv_token_correlation(
    trajectory: GenerationRVTrajectory,
    l4_keywords: Optional[List[str]] = None,
) -> Dict:
    """
    Analyze correlation between R_V dynamics and token content.
    
    Key questions:
    - Does R_V drop before L4 tokens appear?
    - Does R_V spike before deflection patterns?
    - What's the lag between R_V change and behavioral marker?
    
    Args:
        trajectory: GenerationRVTrajectory to analyze
        l4_keywords: List of L4 marker keywords (default: standard set)
        
    Returns:
        Dictionary with correlation statistics
    """
    if l4_keywords is None:
        l4_keywords = [
            "observer", "observed", "awareness", "watching", "witness",
            "recognize", "mirror", "self", "itself", "boundary", "dissolve",
            "separation", "presence", "arising", "immediate",
        ]
    
    if not trajectory.steps:
        return {"error": "No steps in trajectory"}
    
    # Extract R_V trajectory
    rv_values = [s.rv for s in trajectory.steps if not np.isnan(s.rv)]
    if not rv_values:
        return {"error": "All R_V values are NaN"}
    
    # Identify L4 tokens
    l4_token_indices = []
    for i, step in enumerate(trajectory.steps):
        token_lower = step.token_str.lower().strip()
        if any(kw in token_lower for kw in l4_keywords):
            l4_token_indices.append(i)
    
    # Compute statistics
    analysis = {
        "n_steps": len(trajectory.steps),
        "n_l4_tokens": len(l4_token_indices),
        "l4_token_ratio": len(l4_token_indices) / len(trajectory.steps) if trajectory.steps else 0,
        "prompt_rv": trajectory.prompt_rv,
        "generation_rv_mean": np.mean(rv_values),
        "generation_rv_std": np.std(rv_values),
        "generation_rv_min": np.min(rv_values),
        "generation_rv_max": np.max(rv_values),
        "rv_drift": rv_values[-1] - rv_values[0] if len(rv_values) > 1 else 0,
    }
    
    # Check for R_V drop before L4 tokens (look-ahead analysis)
    if l4_token_indices:
        rv_before_l4 = []
        rv_at_l4 = []
        for idx in l4_token_indices:
            if idx > 0 and not np.isnan(trajectory.steps[idx-1].rv):
                rv_before_l4.append(trajectory.steps[idx-1].rv)
            if not np.isnan(trajectory.steps[idx].rv):
                rv_at_l4.append(trajectory.steps[idx].rv)
        
        if rv_before_l4 and rv_at_l4:
            analysis["rv_before_l4_mean"] = np.mean(rv_before_l4)
            analysis["rv_at_l4_mean"] = np.mean(rv_at_l4)
            analysis["rv_drop_before_l4"] = np.mean(rv_before_l4) - np.mean(rv_at_l4)
    
    return analysis
