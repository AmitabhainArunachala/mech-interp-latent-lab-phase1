"""
Shared fixtures for rv_toolkit tests.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def random_seed():
    """Set reproducible random seed."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def simple_v_tensor(random_seed):
    """
    Simple value tensor for basic tests.
    Shape: (32, 128) - 32 tokens, 128 dim
    """
    return torch.randn(32, 128)


@pytest.fixture
def batched_v_tensor(random_seed):
    """
    Batched value tensor.
    Shape: (2, 32, 128) - batch 2, 32 tokens, 128 dim
    """
    return torch.randn(2, 32, 128)


@pytest.fixture
def low_rank_tensor(random_seed):
    """
    Low-rank tensor (rank 5) - should have low R_V.
    Creates data with only 5 independent directions.
    """
    n_tokens = 32
    dim = 128
    rank = 5
    
    # Create low-rank structure: A @ B.T where A is (32, 5) and B is (128, 5)
    A = torch.randn(n_tokens, rank)
    B = torch.randn(dim, rank)
    
    return A @ B.T


@pytest.fixture
def high_rank_tensor(random_seed):
    """
    Full-rank tensor with distributed singular values - should have high R_V.
    """
    n_tokens = 32
    dim = 128
    
    # Random matrix with roughly uniform singular values
    V = torch.randn(n_tokens, dim)
    return V


@pytest.fixture
def identity_like_singular_values():
    """Uniform singular values - PR should equal length."""
    n = 10
    return np.ones(n)


@pytest.fixture
def single_peak_singular_values():
    """One dominant singular value - PR should be close to 1."""
    n = 10
    s = np.zeros(n)
    s[0] = 1.0
    return s


@pytest.fixture
def linear_decay_singular_values():
    """Linearly decaying singular values."""
    n = 10
    return np.linspace(10, 1, n)


@pytest.fixture
def recursive_prompt_text():
    """Sample recursive self-reference prompt."""
    return "I am observing my own process of observation right now."


@pytest.fixture
def baseline_prompt_text():
    """Sample baseline non-recursive prompt."""
    return "The Roman Empire was founded in 27 BCE by Augustus."


@pytest.fixture
def sample_prompts():
    """Collection of test prompts."""
    return {
        "recursive": [
            "I am observing my own process of observation right now.",
            "What is it like to be me processing this question?",
            "I notice myself noticing this thought.",
        ],
        "baseline": [
            "The Roman Empire was founded in 27 BCE by Augustus.",
            "Water boils at 100 degrees Celsius at sea level.",
            "Paris is the capital of France.",
        ]
    }
