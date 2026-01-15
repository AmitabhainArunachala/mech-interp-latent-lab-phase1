"""
Logit Difference: Linear metric for recursive vs baseline attribution.

Per Nanda (2023): "Logit difference is a fantastic metric."

Key property: Linear in residual stream, enabling direct component attribution.

logit_diff = logit(recursive_token) - logit(baseline_token)
"""

from __future__ import annotations

import torch
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass


# Recursive tokens to track
RECURSIVE_TOKENS = [
    "self", "itself", "observer", "awareness", "consciousness",
    "recursive", "loop", "reflection", "witness", "meta",
    "process", "solution", "sentence", "question", "answer",
]

# Baseline/task tokens (common completions)
BASELINE_TOKENS = [
    "the", "a", "an", "is", "are", "was", "were", "be",
    "to", "of", "and", "that", "this", "it", "for", "on",
    "with", "as", "at", "by", "from", "or", "but", "not",
]


@dataclass
class LogitDiffResult:
    """Logit difference result."""
    logit_diff: float  # Main metric: max(recursive) - max(baseline)
    top_recursive_token: str
    top_recursive_logit: float
    top_baseline_token: str
    top_baseline_logit: float
    recursive_mean_logit: float
    baseline_mean_logit: float


class LogitDiffMetric:
    """
    Compute logit difference between recursive and baseline token sets.
    
    Linear in residual stream — suitable for component attribution.
    """
    
    def __init__(
        self,
        tokenizer,
        device: str = "cuda",
        recursive_tokens: Optional[List[str]] = None,
        baseline_tokens: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = tokenizer.vocab_size
        
        # Build token ID sets
        self.recursive_tokens = recursive_tokens or RECURSIVE_TOKENS
        self.baseline_tokens = baseline_tokens or BASELINE_TOKENS
        
        self.recursive_ids = self._tokens_to_ids(self.recursive_tokens)
        self.baseline_ids = self._tokens_to_ids(self.baseline_tokens)
        
        print(f"[LogitDiffMetric] Recursive tokens: {len(self.recursive_ids)} IDs")
        print(f"[LogitDiffMetric] Baseline tokens: {len(self.baseline_ids)} IDs")
    
    def _tokens_to_ids(self, tokens: List[str]) -> torch.Tensor:
        """Convert token strings to IDs, handling variations."""
        ids = set()
        
        for token in tokens:
            # Try multiple encodings
            variations = [
                token,
                " " + token,
                token.capitalize(),
                " " + token.capitalize(),
            ]
            
            for v in variations:
                try:
                    encoded = self.tokenizer.encode(v, add_special_tokens=False)
                    if len(encoded) == 1:
                        ids.add(encoded[0])
                except Exception:
                    continue
        
        return torch.tensor(sorted(ids), device=self.device, dtype=torch.long)
    
    def compute(
        self,
        logits: torch.Tensor,
        position: int = -1,
    ) -> LogitDiffResult:
        """
        Compute logit difference at a specific position.
        
        Args:
            logits: (seq_len, vocab_size) or (batch, seq_len, vocab_size)
            position: Which position to analyze (default: -1 = last)
        
        Returns:
            LogitDiffResult with the logit difference and details
        """
        # Handle dimensions
        if logits.dim() == 3:
            logits = logits[0]  # Take first batch
        
        # Get logits at target position
        pos_logits = logits[position]  # (vocab_size,)
        
        # Extract recursive and baseline logits
        r_logits = pos_logits[self.recursive_ids]
        b_logits = pos_logits[self.baseline_ids]
        
        # Compute max and mean
        r_max, r_max_idx = r_logits.max(dim=0)
        b_max, b_max_idx = b_logits.max(dim=0)
        
        r_mean = r_logits.mean()
        b_mean = b_logits.mean()
        
        # Logit difference (linear!)
        logit_diff = (r_max - b_max).item()
        
        # Decode top tokens
        top_r_id = self.recursive_ids[r_max_idx].item()
        top_b_id = self.baseline_ids[b_max_idx].item()
        
        return LogitDiffResult(
            logit_diff=logit_diff,
            top_recursive_token=self.tokenizer.decode([top_r_id]),
            top_recursive_logit=r_max.item(),
            top_baseline_token=self.tokenizer.decode([top_b_id]),
            top_baseline_logit=b_max.item(),
            recursive_mean_logit=r_mean.item(),
            baseline_mean_logit=b_mean.item(),
        )
    
    def compute_trajectory(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        model,
        position: int = -1,
    ) -> List[LogitDiffResult]:
        """
        Compute logit difference at each layer (logit lens style).
        
        This shows WHERE in the network the model starts preferring
        recursive tokens over baseline tokens.
        
        Args:
            hidden_states: From model(..., output_hidden_states=True)
            model: Model with model.model.norm and model.lm_head
            position: Position to analyze
        
        Returns:
            List of LogitDiffResult, one per layer
        """
        results = []
        
        for layer_idx, h in enumerate(hidden_states):
            # Apply final LayerNorm + LM head (logit lens)
            h_pos = h[0, position, :]
            h_norm = model.model.norm(h_pos)
            logits = model.lm_head(h_norm).unsqueeze(0)  # (1, vocab_size)
            
            result = self.compute(logits, position=0)
            results.append(result)
        
        return results
    
    def find_crossover_layer(
        self,
        trajectory: List[LogitDiffResult],
    ) -> Optional[int]:
        """
        Find the layer where logit_diff crosses from negative to positive.
        
        This is the layer where the model "decides" this is recursive.
        
        Returns:
            Layer index of first positive logit_diff, or None if never positive
        """
        for i, result in enumerate(trajectory):
            if result.logit_diff > 0:
                return i
        return None
