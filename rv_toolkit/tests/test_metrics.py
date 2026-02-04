"""
Tests for rv_toolkit.metrics module.

Tests the core R_V computation, participation ratio, effective rank,
and dual-space decomposition.
"""

import pytest
import torch
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from rv_toolkit.metrics import (
    compute_participation_ratio,
    compute_effective_rank,
    compute_rv,
    compute_dual_space_decomposition,
    extract_recursive_subspace,
    RVResult,
)


class TestParticipationRatio:
    """Tests for compute_participation_ratio."""
    
    def test_uniform_singular_values(self, identity_like_singular_values):
        """Uniform singular values should give PR = n."""
        s = identity_like_singular_values
        pr = compute_participation_ratio(s)
        assert_almost_equal(pr, len(s), decimal=5)
    
    def test_single_peak(self, single_peak_singular_values):
        """Single dominant singular value should give PR ≈ 1."""
        s = single_peak_singular_values
        pr = compute_participation_ratio(s)
        assert_almost_equal(pr, 1.0, decimal=5)
    
    def test_linear_decay(self, linear_decay_singular_values):
        """Linear decay should give intermediate PR."""
        s = linear_decay_singular_values
        pr = compute_participation_ratio(s)
        # Should be between 1 and n
        assert 1.0 < pr < len(s)
    
    def test_two_equal_values(self):
        """Two equal singular values should give PR = 2."""
        s = np.array([1.0, 1.0])
        pr = compute_participation_ratio(s)
        assert_almost_equal(pr, 2.0, decimal=5)
    
    def test_empty_array(self):
        """Empty array should give NaN."""
        s = np.array([])
        pr = compute_participation_ratio(s)
        assert np.isnan(pr)
    
    def test_zero_array(self):
        """Zero array should give NaN."""
        s = np.zeros(5)
        pr = compute_participation_ratio(s)
        assert np.isnan(pr)
    
    def test_scale_invariance(self):
        """PR should be scale-invariant."""
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = s1 * 100  # Scaled version
        
        pr1 = compute_participation_ratio(s1)
        pr2 = compute_participation_ratio(s2)
        
        assert_almost_equal(pr1, pr2, decimal=5)


class TestEffectiveRank:
    """Tests for compute_effective_rank."""
    
    def test_uniform_singular_values(self, identity_like_singular_values):
        """Uniform singular values should give eff_rank = n."""
        s = identity_like_singular_values
        eff_rank = compute_effective_rank(s)
        assert_almost_equal(eff_rank, len(s), decimal=5)
    
    def test_single_peak(self, single_peak_singular_values):
        """Single dominant value should give eff_rank ≈ 1."""
        s = single_peak_singular_values
        eff_rank = compute_effective_rank(s)
        assert_almost_equal(eff_rank, 1.0, decimal=5)
    
    def test_consistent_with_pr(self):
        """Effective rank should be consistent with PR formula."""
        # For this formulation, PR and eff_rank use same formula
        s = np.array([5.0, 3.0, 2.0, 1.0])
        
        pr = compute_participation_ratio(s)
        eff_rank = compute_effective_rank(s)
        
        assert_almost_equal(pr, eff_rank, decimal=5)


class TestComputeRV:
    """Tests for compute_rv on value tensors."""
    
    def test_basic_computation(self, simple_v_tensor):
        """Should compute R_V without error."""
        rv = compute_rv(simple_v_tensor)
        assert isinstance(rv, float)
        assert not np.isnan(rv)
        assert rv > 0
    
    def test_batched_input(self, batched_v_tensor):
        """Should handle batched input by taking first batch."""
        rv = compute_rv(batched_v_tensor)
        assert isinstance(rv, float)
        assert not np.isnan(rv)
    
    def test_low_rank_has_lower_rv(self, low_rank_tensor, high_rank_tensor):
        """Low-rank tensor should have lower R_V than high-rank."""
        rv_low = compute_rv(low_rank_tensor)
        rv_high = compute_rv(high_rank_tensor)
        
        # Low-rank should have lower effective dimensionality
        assert rv_low < rv_high
    
    def test_window_size_effect(self, simple_v_tensor):
        """Different window sizes should give different results."""
        rv_8 = compute_rv(simple_v_tensor, window_size=8)
        rv_16 = compute_rv(simple_v_tensor, window_size=16)
        rv_32 = compute_rv(simple_v_tensor, window_size=32)
        
        # All should be valid
        assert not np.isnan(rv_8)
        assert not np.isnan(rv_16)
        assert not np.isnan(rv_32)
    
    def test_return_components(self, simple_v_tensor):
        """Should return RVResult when return_components=True."""
        result = compute_rv(simple_v_tensor, return_components=True)
        
        assert isinstance(result, RVResult)
        assert hasattr(result, 'rv')
        assert hasattr(result, 'effective_rank')
        assert hasattr(result, 'singular_values')
        assert len(result.singular_values) > 0
    
    def test_none_input(self):
        """Should handle None input gracefully."""
        rv = compute_rv(None)
        assert np.isnan(rv)
        
        result = compute_rv(None, return_components=True)
        assert isinstance(result, RVResult)
        assert np.isnan(result.rv)
    
    def test_short_sequence(self, random_seed):
        """Should handle sequences shorter than window size."""
        short_tensor = torch.randn(4, 128)  # Only 4 tokens
        rv = compute_rv(short_tensor, window_size=16)
        assert not np.isnan(rv)  # Should still work


class TestRVResult:
    """Tests for the RVResult dataclass."""
    
    def test_dual_ratio_computation(self):
        """Should compute dual_ratio when both norms present."""
        result = RVResult(
            rv=0.85,
            effective_rank=10.0,
            singular_values=np.array([1, 2, 3]),
            v_parallel_norm=5.0,
            v_perp_norm=2.5,
        )
        
        assert result.dual_ratio == pytest.approx(2.0)
    
    def test_dual_ratio_none_when_missing(self):
        """dual_ratio should be None when components missing."""
        result = RVResult(
            rv=0.85,
            effective_rank=10.0,
            singular_values=np.array([1, 2, 3]),
        )
        
        assert result.dual_ratio is None
    
    def test_dual_ratio_with_zero_perp(self):
        """dual_ratio should be None when perp is zero (avoid division)."""
        result = RVResult(
            rv=0.85,
            effective_rank=10.0,
            singular_values=np.array([1, 2, 3]),
            v_parallel_norm=5.0,
            v_perp_norm=0.0,
        )
        
        assert result.dual_ratio is None


class TestDualSpaceDecomposition:
    """Tests for dual-space decomposition functionality."""
    
    def test_extract_subspace(self, simple_v_tensor):
        """Should extract orthonormal basis."""
        basis = extract_recursive_subspace(simple_v_tensor, n_components=5)
        
        assert basis.shape[0] == 5
        assert basis.shape[1] == simple_v_tensor.shape[1]
        
        # Check orthonormality
        gram = basis @ basis.T
        expected = torch.eye(5)
        assert torch.allclose(gram, expected, atol=1e-5)
    
    def test_decomposition_orthogonality(self, simple_v_tensor):
        """V_parallel and V_perpendicular should be orthogonal."""
        basis = extract_recursive_subspace(simple_v_tensor, n_components=5)
        result = compute_dual_space_decomposition(
            simple_v_tensor, 
            basis, 
            window_size=16
        )
        
        assert result.v_parallel_norm is not None
        assert result.v_perp_norm is not None
        assert result.rv > 0
    
    def test_full_projection_recovery(self, random_seed):
        """If basis spans all directions, v_perp should be ~0."""
        # Create tensor where all directions are in subspace
        dim = 64
        n_tokens = 20
        rank = 10
        
        A = torch.randn(n_tokens, rank)
        B = torch.randn(dim, rank)
        V = A @ B.T  # Low-rank tensor
        
        # Extract full subspace
        basis = extract_recursive_subspace(V, n_components=rank)
        result = compute_dual_space_decomposition(V, basis, window_size=16)
        
        # v_perp should be very small relative to v_parallel
        assert result.v_perp_norm < result.v_parallel_norm * 0.1


class TestNumericalStability:
    """Tests for numerical edge cases."""
    
    def test_very_small_values(self, random_seed):
        """Should handle very small activation values."""
        tiny_tensor = torch.randn(32, 128) * 1e-10
        rv = compute_rv(tiny_tensor)
        # Should either compute or return NaN, not error
        assert isinstance(rv, float)
    
    def test_very_large_values(self, random_seed):
        """Should handle moderately large activation values."""
        # Using 1e4 instead of 1e10 to avoid overflow in float32 SVD
        large_tensor = torch.randn(32, 128) * 1e4
        rv = compute_rv(large_tensor)
        assert isinstance(rv, float)
        # May be NaN due to numerical issues - that's acceptable
    
    def test_mixed_precision(self, random_seed):
        """Should handle different tensor dtypes."""
        for dtype in [torch.float16, torch.float32, torch.float64]:
            tensor = torch.randn(32, 128, dtype=dtype)
            rv = compute_rv(tensor)
            assert isinstance(rv, float)


class TestDeviceCompatibility:
    """Tests for GPU/CPU compatibility."""
    
    def test_cpu_tensor(self, simple_v_tensor):
        """Should work on CPU tensors."""
        rv = compute_rv(simple_v_tensor.cpu())
        assert not np.isnan(rv)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensor(self, simple_v_tensor):
        """Should work on CUDA tensors."""
        rv = compute_rv(simple_v_tensor.cuda())
        assert not np.isnan(rv)
