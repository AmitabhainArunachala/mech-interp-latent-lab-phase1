"""
Mode Score M Metric.

Defines the logit-level signature of "Recursive Mode".
M = logsumexp(logits[Recursive]) - logsumexp(logits[Task])

Recursive tokens (R):
- observer, observed, awareness, itself, self, recognition, consciousness, witness, reflection
- Plus variants (case, spacing, plurals)

Task tokens (T):
- Top-K from clean baseline logits (dynamic) OR fixed domain tokens.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Set, Dict

# Base recursive keywords to expand into tokens
RECURSIVE_KEYWORDS = [
    "observer", "observed", "awareness", "itself", "self", "recognition",
    "consciousness", "witness", "reflection", "recursive", "loop", "meta",
    "observing", "watching", "monitoring", "knowing", "relating", "relates",
    "relate", "themselves", "yourself", "myself", "conscious", "aware",
    "self-aware", "self-awareness", "self-reference", "self-referential",
    "introspection", "introspective", "metacognition", "metacognitive"
]

class ModeScoreMetric:
    def __init__(self, tokenizer, device="cuda"):
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else tokenizer.vocab_size
        self.recursive_token_ids = self._build_recursive_set()
        
    def _build_recursive_set(self) -> torch.Tensor:
        """
        Identify token IDs that correspond to recursive concepts.
        Includes comprehensive variations: case, spacing, plurals, compounds.
        
        Returns:
            torch.Tensor of token IDs (shape: [n_tokens])
        """
        ids = set()
        
        # Generate variations for each keyword
        for word in RECURSIVE_KEYWORDS:
            variations = [
                word,
                " " + word,
                word.capitalize(),
                " " + word.capitalize(),
                word.upper(),
                " " + word.upper(),
                word + "s",  # plural
                " " + word + "s",
                word + "ing",  # gerund
                " " + word + "ing",
            ]
            
            # Add hyphenated variants for compound words
            if "-" not in word:
                variations.extend([
                    word + "-aware",
                    word + "-reference",
                    word + "-referential",
                ])
            
            for v in variations:
                try:
                    tokens = self.tokenizer.encode(v, add_special_tokens=False)
                    # Prefer single tokens, but also include multi-token words
                    # if they're common recursive concepts
                    if len(tokens) == 1:
                        ids.add(tokens[0])
                    elif len(tokens) == 2 and any(
                        kw in v.lower() for kw in ["consciousness", "awareness", "observer"]
                    ):
                        # Include both tokens for important multi-token concepts
                        ids.update(tokens)
                except Exception:
                    continue
        
        # Validate token IDs are within vocab size
        valid_ids = [tid for tid in ids if 0 <= tid < self.vocab_size]
        
        if len(valid_ids) == 0:
            raise ValueError("No valid recursive tokens found!")
        
        token_tensor = torch.tensor(sorted(valid_ids), device=self.device, dtype=torch.long)
        
        # Print summary for validation
        print(f"[ModeScoreMetric] Built recursive token set: {len(token_tensor)} tokens")
        print(f"[ModeScoreMetric] Token ID range: {token_tensor.min().item()} - {token_tensor.max().item()}")
        
        return token_tensor
    
    def get_task_token_set(
        self,
        baseline_logits: torch.Tensor,
        top_k: int = 10,
        per_position: bool = True
    ) -> torch.Tensor:
        """
        Extract task token set T from baseline logits.
        
        Args:
            baseline_logits: (seq_len, vocab_size) or (batch, seq_len, vocab_size)
            top_k: Number of top tokens to include per position
            per_position: If True, return per-position indices. If False, return union.
        
        Returns:
            If per_position: (seq_len, top_k) tensor of token indices
            If not per_position: (n_tokens,) tensor of unique token indices
        """
        if baseline_logits.dim() == 3:
            baseline_logits = baseline_logits[0]  # Take first batch
        
        seq_len, vocab_size = baseline_logits.shape
        
        if per_position:
            # Top-K per position
            _, task_indices = torch.topk(baseline_logits, k=top_k, dim=-1)  # (seq_len, top_k)
            return task_indices
        else:
            # Union of top-K across all positions
            all_indices = set()
            for pos in range(seq_len):
                _, top_indices = torch.topk(baseline_logits[pos], k=top_k)
                all_indices.update(top_indices.cpu().tolist())
            return torch.tensor(sorted(all_indices), device=self.device, dtype=torch.long)
    
    def validate_token_sets(self, sample_prompt: str = "What is consciousness?"):
        """
        Validate token sets by printing examples and checking coverage.
        
        Args:
            sample_prompt: Sample prompt to test on
        """
        print("\n=== ModeScoreMetric Validation ===")
        print(f"Recursive token set size: {len(self.recursive_token_ids)}")
        
        # Decode some recursive tokens to verify
        sample_ids = self.recursive_token_ids[:10].cpu().tolist()
        sample_tokens = [self.tokenizer.decode([tid]) for tid in sample_ids]
        print(f"Sample recursive tokens: {sample_tokens[:5]}")
        
        # Test on sample prompt
        sample_tokens_encoded = self.tokenizer.encode(sample_prompt, add_special_tokens=False)
        print(f"\nSample prompt: '{sample_prompt}'")
        print(f"Encoded as {len(sample_tokens_encoded)} tokens")
        
        # Check overlap
        recursive_set = set(self.recursive_token_ids.cpu().tolist())
        prompt_set = set(sample_tokens_encoded)
        overlap = recursive_set & prompt_set
        print(f"Overlap with recursive set: {len(overlap)} tokens")
        
        print("=" * 40)

    def compute_score(
        self,
        logits: torch.Tensor,
        baseline_logits: Optional[torch.Tensor] = None,
        top_k_task: int = 10,
        per_position: bool = True
    ) -> float:
        """
        Compute Mode Score M for logits.
        
        M = logsumexp(logits[R]) - logsumexp(logits[T])
        
        Args:
            logits: (seq_len, vocab_size) or (batch, seq_len, vocab_size)
            baseline_logits: (seq_len, vocab_size) - used to define T (Task tokens).
                            If None, returns LSE(R) only (raw recursive strength).
            top_k_task: Number of top tokens to use for task set T
            per_position: If True, compute T per position. If False, use union.
        
        Returns:
            Mean Mode Score M over sequence (float)
        """
        original_shape = logits.shape
        
        # Handle batch dimension
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)  # (batch*seq_len, vocab_size)
            if baseline_logits is not None:
                if baseline_logits.dim() == 3:
                    baseline_logits = baseline_logits.view(-1, vocab_size)
                else:
                    # Broadcast baseline to match logits
                    baseline_logits = baseline_logits.unsqueeze(0).expand(batch_size, -1, -1).view(-1, vocab_size)
        else:
            seq_len, vocab_size = logits.shape
        
        # 1. Recursive Logsumexp
        # Extract logits for recursive tokens
        r_logits = logits[:, self.recursive_token_ids]  # (N, n_recursive_tokens)
        lse_r = torch.logsumexp(r_logits, dim=-1)  # (N,)
        
        # 2. Task Logsumexp
        if baseline_logits is not None:
            if per_position:
                # Top-K per position
                _, task_indices = torch.topk(baseline_logits, k=top_k_task, dim=-1)  # (N, top_k)
                # Gather task logits from current logits
                t_logits = torch.gather(logits, dim=-1, index=task_indices)  # (N, top_k)
                lse_t = torch.logsumexp(t_logits, dim=-1)  # (N,)
            else:
                # Union of top-K across positions
                task_set = self.get_task_token_set(baseline_logits, top_k=top_k_task, per_position=False)
                t_logits = logits[:, task_set]  # (N, n_task_tokens)
                lse_t = torch.logsumexp(t_logits, dim=-1)  # (N,)
        else:
            # Fallback: Just return LSE(R) (raw recursive strength)
            lse_t = torch.zeros_like(lse_r)
        
        # M = LSE(R) - LSE(T)
        m = lse_r - lse_t
        
        # Return mean over sequence
        return m.mean().item()
    
    def compute_score_per_step(
        self,
        logits: torch.Tensor,
        baseline_logits: Optional[torch.Tensor] = None,
        top_k_task: int = 10,
        n_steps: Optional[int] = None
    ) -> List[float]:
        """
        Compute Mode Score M per generation step.
        
        Args:
            logits: (seq_len, vocab_size) or (batch, seq_len, vocab_size)
            baseline_logits: (seq_len, vocab_size) - used to define T
            top_k_task: Number of top tokens for task set
            n_steps: Number of steps to compute (default: all steps)
        
        Returns:
            List of M scores per step
        """
        if logits.dim() == 3:
            logits = logits[0]  # Take first batch
        
        seq_len = logits.shape[0]
        if n_steps is not None:
            seq_len = min(seq_len, n_steps)
        
        scores = []
        for step in range(seq_len):
            step_logits = logits[step:step+1]  # (1, vocab_size)
            if baseline_logits is not None:
                step_baseline = baseline_logits[step:step+1] if baseline_logits.shape[0] > step else baseline_logits[-1:]
            else:
                step_baseline = None
            
            m = self.compute_score(step_logits, step_baseline, top_k_task=top_k_task)
            scores.append(m)
        
        return scores


