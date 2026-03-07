"""R_V measurement utilities.

R_V = PR_late / PR_early

Where PR (Participation Ratio) = (Σλᵢ²)² / Σλᵢ⁴
Computed via SVD on Value matrix column space.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple


def compute_participation_ratio(v_tensor: torch.Tensor, window_size: int = None) -> float:
    """Compute participation ratio from Value matrix.

    Args:
        v_tensor: Tensor of shape [seq_len, hidden] or [batch, seq_len, hidden]
        window_size: If specified, use only last N tokens

    Returns:
        Participation ratio (float)
    """
    if v_tensor is None or v_tensor.numel() == 0:
        return np.nan

    # Handle batch dimension
    if len(v_tensor.shape) == 3:
        v_tensor = v_tensor[0]  # Take first batch

    # Extract window if specified
    if window_size is not None:
        seq_len = v_tensor.shape[0]
        start_idx = max(0, seq_len - window_size)
        v_tensor = v_tensor[start_idx:, :]

    # Check minimum size
    if v_tensor.shape[0] < 2:
        return np.nan

    # SVD
    try:
        # Move to CPU for SVD if needed
        v_np = v_tensor.detach().cpu().float().numpy()
        U, S, Vt = np.linalg.svd(v_np, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.nan

    # Squared singular values
    S_sq = S ** 2

    # Participation ratio: (sum(s_i^2))^2 / sum(s_i^4)
    numerator = (S_sq.sum()) ** 2
    denominator = ((S_sq ** 2).sum())

    if denominator == 0:
        return np.nan

    participation_ratio = numerator / denominator

    return float(participation_ratio)


def extract_value_matrix(model, inputs: Dict, layer_idx: int) -> torch.Tensor:
    """Extract Value matrix from a specific layer.

    Args:
        model: Transformer model
        inputs: Dict with 'input_ids' (and optionally 'attention_mask')
        layer_idx: Which layer to extract from (0-indexed)

    Returns:
        Value tensor of shape [seq_len, hidden]
    """
    # Run forward pass with output_hidden_states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # Extract hidden state at this layer
    # hidden_states is tuple of (num_layers + 1) tensors
    # Index 0 is embeddings, 1 is layer 0, etc.
    hidden_state = outputs.hidden_states[layer_idx + 1]  # +1 for embeddings

    # hidden_state shape: [batch, seq_len, hidden]
    # For Value matrix, we use the hidden state directly
    # (In attention, V = hidden_state @ W_V, but for PR we can use hidden_state)

    return hidden_state[0]  # Return first batch, shape [seq_len, hidden]


def measure_r_v_single_prompt(
    model,
    inputs: Dict,
    early_layers: List[int] = [1, 2, 3, 4, 5],
    late_layers: List[int] = [20, 21, 22, 23, 24, 25, 26, 27],
    window_size: int = None,
) -> float:
    """Measure R_V for a single prompt.

    R_V = PR_late / PR_early

    Args:
        model: Transformer model
        inputs: Dict with 'input_ids' (and optionally 'attention_mask')
        early_layers: List of early layer indices
        late_layers: List of late layer indices
        window_size: If specified, use only last N tokens for PR computation

    Returns:
        R_V ratio (float)
    """
    # Compute PR for early layers (average across layers)
    pr_early_values = []
    for layer_idx in early_layers:
        try:
            v_tensor = extract_value_matrix(model, inputs, layer_idx)
            pr = compute_participation_ratio(v_tensor, window_size=window_size)
            if not np.isnan(pr):
                pr_early_values.append(pr)
        except Exception:
            continue

    # Compute PR for late layers (average across layers)
    pr_late_values = []
    for layer_idx in late_layers:
        try:
            v_tensor = extract_value_matrix(model, inputs, layer_idx)
            pr = compute_participation_ratio(v_tensor, window_size=window_size)
            if not np.isnan(pr):
                pr_late_values.append(pr)
        except Exception:
            continue

    # Average
    if not pr_early_values or not pr_late_values:
        return np.nan

    pr_early = sum(pr_early_values) / len(pr_early_values)
    pr_late = sum(pr_late_values) / len(pr_late_values)

    # R_V ratio
    if pr_early == 0:
        return np.nan

    r_v = pr_late / pr_early

    return float(r_v)


def measure_r_v_at_layers(
    model,
    inputs: Dict,
    early_layers: List[int] = [1, 2, 3, 4, 5],
    late_layers: List[int] = [20, 21, 22, 23, 24, 25, 26, 27],
) -> float:
    """Alias for measure_r_v_single_prompt with standard layer groups.

    This is the function signature expected by multi_token_r_v_experiment.py
    """
    return measure_r_v_single_prompt(
        model=model,
        inputs=inputs,
        early_layers=early_layers,
        late_layers=late_layers,
    )


if __name__ == "__main__":
    # Test with a small model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Testing R_V measurement...")

    model_name = "gpt2"  # Small model for testing
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test prompt
    prompt = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(prompt, return_tensors="pt")

    # Measure R_V
    r_v = measure_r_v_single_prompt(
        model=model,
        inputs=inputs,
        early_layers=[0, 1, 2],
        late_layers=[9, 10, 11],  # GPT-2 has 12 layers
    )

    print(f"Test prompt: {prompt}")
    print(f"R_V: {r_v:.4f}")

    if not np.isnan(r_v):
        print("✓ R_V measurement working")
    else:
        print("✗ R_V measurement failed")
