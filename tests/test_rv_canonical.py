"""
Phase 0 canonicalization gate: metric equivalence tests for geometric_lens/metrics.py.

Proves that the canonical R_V implementation computes the correct values against
known analytic inputs, handles all edge cases without crashing, and agrees with
the legacy implementation in src/metrics/rv.py.

Paper formula (all TeX versions agree):
    PR(l) = (sum_i sigma_i^2)^2 / sum_i sigma_i^4

Where sigma_i are singular values of V^(l).T (the transposed activation window).
R_V = PR(late) / PR(early).

Note on formula variant: The user prompt mentions PR = (sum sigma_i)^2 / sum sigma_i^2
(operating on raw singular values). The paper and code both use the SQUARED singular
value form: PR = (sum sigma_i^2)^2 / sum sigma_i^4, which is the standard participation
ratio from condensed matter physics applied to eigenvalues (= sigma^2). These tests
verify the code matches the paper.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch

# Ensure the project root is on sys.path so geometric_lens and src are importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometric_lens.metrics import (
    compute_rv,
    compute_rv_with_components,
    participation_ratio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pr_reference(sigma: np.ndarray) -> float:
    """Reference PR implementation from first principles matching the paper.

    PR = (sum(sigma_i^2))^2 / sum(sigma_i^4)

    This is the formula in every version of the R_V paper (paper.tex,
    paper_colm2026.tex, paper_colm2026_v005.tex, DRAFT_SECTIONS_1_3).
    """
    s2 = sigma ** 2
    numerator = s2.sum() ** 2
    denominator = (s2 ** 2).sum()
    if denominator < 1e-30:
        return float("nan")
    return float(numerator / denominator)


def _make_tensor_with_known_svd(
    singular_values: list[float],
    seq_len: int = 16,
    dim: int = 64,
) -> torch.Tensor:
    """Build a (seq_len, dim) tensor whose SVD of .T has the given singular values.

    Strategy: construct M = U @ diag(s) @ Vt where U and Vt are truncated
    orthogonal matrices, then return M.T transposed back so that the code's
    internal `v_cpu.T` recovers the intended spectrum.

    The code does: SVD(tensor[-W:, :].T)  [shape (dim, W)]
    So we need tensor[-W:, :] such that its transpose has the desired sigmas.
    That means tensor[-W:, :] = (U @ diag(s) @ Vt).T = Vt.T @ diag(s) @ U.T
    Equivalently we can just set tensor[-W:, :] = V @ diag(s) @ U.T
    where U is (dim, k) and V is (W, k) orthonormal columns.
    """
    k = len(singular_values)
    assert k <= min(seq_len, dim), "Too many singular values for the given shape"

    rng = np.random.RandomState(42)

    # Random orthogonal bases via QR
    A = rng.randn(dim, k)
    U, _ = np.linalg.qr(A)  # (dim, k) orthonormal columns

    B = rng.randn(seq_len, k)
    V, _ = np.linalg.qr(B)  # (seq_len, k) orthonormal columns

    S = np.diag(singular_values)

    # We want SVD(tensor.T) to yield these singular values.
    # tensor.T has shape (dim, seq_len).
    # Let tensor.T = U @ S @ V.T  =>  tensor = V @ S @ U.T
    tensor_np = V @ S @ U.T  # (seq_len, dim)

    return torch.tensor(tensor_np, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_16x64() -> torch.Tensor:
    """A (16, 64) tensor whose transpose has 16 equal singular values of 1.0.

    This is the identity-like case: PR should equal 16 exactly.
    """
    return _make_tensor_with_known_svd([1.0] * 16, seq_len=16, dim=64)


@pytest.fixture
def rank1_16x64() -> torch.Tensor:
    """A (16, 64) rank-1 tensor. PR should equal 1.0 exactly."""
    return _make_tensor_with_known_svd([5.0], seq_len=16, dim=64)


@pytest.fixture
def known_spectrum_tensor() -> torch.Tensor:
    """A tensor with singular values [4.0, 2.0, 1.0]. PR is analytically known."""
    return _make_tensor_with_known_svd([4.0, 2.0, 1.0], seq_len=16, dim=64)


# ---------------------------------------------------------------------------
# Test Class 1: PR formula correctness
# ---------------------------------------------------------------------------

class TestParticipationRatioFormula:
    """Verify that participation_ratio computes the paper's formula exactly."""

    def test_pr_identity_equals_k(self, identity_16x64: torch.Tensor) -> None:
        """PR of a matrix with k equal singular values must equal k.

        Proof: if all sigma_i = c, then
            PR = (k * c^2)^2 / (k * c^4) = k^2 * c^4 / (k * c^4) = k.
        """
        pr = participation_ratio(identity_16x64, window_size=16)
        assert pr == pytest.approx(16.0, abs=1e-6), (
            f"PR of identity-like matrix should be 16.0, got {pr}"
        )

    def test_pr_rank1_equals_one(self, rank1_16x64: torch.Tensor) -> None:
        """PR of a rank-1 matrix must equal 1.0 exactly.

        Proof: only one nonzero sigma, so
            PR = (sigma^2)^2 / (sigma^4) = 1.0.
        """
        pr = participation_ratio(rank1_16x64, window_size=16)
        assert pr == pytest.approx(1.0, abs=1e-6), (
            f"PR of rank-1 matrix should be 1.0, got {pr}"
        )

    def test_pr_known_spectrum(self, known_spectrum_tensor: torch.Tensor) -> None:
        """PR for sigma = [4, 2, 1] should match the analytic value.

        sigma^2 = [16, 4, 1], sum = 21, sum^2 = 441
        sigma^4 = [256, 16, 1], sum = 273
        PR = 441 / 273 = 1.615384615...
        """
        expected = 441.0 / 273.0
        pr = participation_ratio(known_spectrum_tensor, window_size=16)
        assert pr == pytest.approx(expected, abs=1e-4), (
            f"PR for sigma=[4,2,1] should be {expected:.6f}, got {pr}"
        )

    def test_pr_matches_reference_implementation(self) -> None:
        """Compare canonical PR against a pure-numpy reference over random spectra.

        Uses 50 random spectra to verify the code path agrees with the paper
        formula implemented independently in _pr_reference().
        """
        rng = np.random.RandomState(123)
        for _ in range(50):
            k = rng.randint(2, 16)
            sigmas = rng.uniform(0.1, 10.0, size=k).tolist()
            tensor = _make_tensor_with_known_svd(sigmas, seq_len=16, dim=64)
            pr_code = participation_ratio(tensor, window_size=16)
            pr_ref = _pr_reference(np.array(sigmas))
            assert pr_code == pytest.approx(pr_ref, rel=1e-4), (
                f"Mismatch for sigmas={sigmas}: code={pr_code}, ref={pr_ref}"
            )

    def test_pr_scale_invariance(self) -> None:
        """PR is scale-invariant: scaling all activations by c does not change PR.

        Proof: if sigma -> c*sigma, then
            PR = (sum (c*sigma_i)^2)^2 / sum (c*sigma_i)^4
               = c^4 * (sum sigma_i^2)^2 / (c^8 ... no, let's just test it.

        Actually: sigma^2 -> c^2 * sigma^2, sigma^4 -> c^4 * sigma^4.
            PR = (c^2 * sum s^2)^2 / (c^4 * sum s^4) = c^4 / c^4 * original = original.
        """
        base = _make_tensor_with_known_svd([3.0, 2.0, 1.0], seq_len=16, dim=64)
        scaled = base * 1000.0

        pr_base = participation_ratio(base, window_size=16)
        pr_scaled = participation_ratio(scaled, window_size=16)
        assert pr_base == pytest.approx(pr_scaled, rel=1e-5)

    def test_pr_two_equal_singular_values(self) -> None:
        """Two equal singular values should give PR = 2.0."""
        tensor = _make_tensor_with_known_svd([3.0, 3.0], seq_len=16, dim=64)
        pr = participation_ratio(tensor, window_size=16)
        assert pr == pytest.approx(2.0, abs=1e-6)

    def test_pr_bounded_between_1_and_k(self) -> None:
        """PR is always in [1, k] where k = min(seq_len, dim) nonzero sigmas."""
        rng = np.random.RandomState(77)
        for _ in range(30):
            k = rng.randint(2, 16)
            sigmas = rng.uniform(0.01, 10.0, size=k).tolist()
            tensor = _make_tensor_with_known_svd(sigmas, seq_len=16, dim=64)
            pr = participation_ratio(tensor, window_size=16)
            assert 1.0 - 1e-6 <= pr <= k + 1e-6, (
                f"PR={pr} out of bounds [1, {k}] for sigmas={sigmas}"
            )


# ---------------------------------------------------------------------------
# Test Class 2: Edge cases
# ---------------------------------------------------------------------------

class TestParticipationRatioEdgeCases:
    """Verify that participation_ratio handles degenerate inputs gracefully."""

    def test_none_input(self) -> None:
        """None input must return NaN, not raise."""
        result = participation_ratio(None)
        assert math.isnan(result)

    def test_all_zeros(self) -> None:
        """All-zero matrix has no meaningful spectrum. Must return NaN."""
        zeros = torch.zeros(16, 64)
        result = participation_ratio(zeros, window_size=16)
        assert math.isnan(result), f"Expected NaN for zero matrix, got {result}"

    def test_sequence_too_short(self) -> None:
        """If T < window_size, must return NaN per measurement contract."""
        short = torch.randn(4, 64)
        result = participation_ratio(short, window_size=16)
        assert math.isnan(result), (
            "Sequence shorter than window_size must return NaN"
        )

    def test_batched_input_takes_first(self) -> None:
        """3D input (batch, seq, dim) should use batch[0] only."""
        rng = torch.Generator().manual_seed(42)
        single = torch.randn(16, 64, generator=rng)

        rng2 = torch.Generator().manual_seed(42)
        batched = torch.randn(1, 16, 64, generator=rng2)

        pr_single = participation_ratio(single, window_size=16)
        pr_batched = participation_ratio(batched, window_size=16)
        assert pr_single == pytest.approx(pr_batched, rel=1e-10)

    def test_nan_in_tensor(self) -> None:
        """Tensor containing NaN values must return NaN, not crash."""
        t = torch.randn(16, 64)
        t[5, 10] = float("nan")
        result = participation_ratio(t, window_size=16)
        assert math.isnan(result)

    def test_inf_in_tensor(self) -> None:
        """Tensor containing Inf values must return NaN, not crash."""
        t = torch.randn(16, 64)
        t[0, 0] = float("inf")
        result = participation_ratio(t, window_size=16)
        assert math.isnan(result)

    def test_single_token_window(self) -> None:
        """Window size 1 means a single row. SVD of a 1-row matrix has 1 SV.

        Result should be PR = 1.0 (one singular value => rank 1).
        """
        t = torch.randn(16, 64)
        result = participation_ratio(t, window_size=1)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_float16_input(self) -> None:
        """float16 input should not crash (code casts to float64 internally)."""
        t = torch.randn(16, 64).half()
        result = participation_ratio(t, window_size=16)
        assert isinstance(result, float)
        assert not math.isnan(result)

    def test_bfloat16_input(self) -> None:
        """bfloat16 is the mandatory dtype for real models. Must not crash."""
        t = torch.randn(16, 64).bfloat16()
        result = participation_ratio(t, window_size=16)
        assert isinstance(result, float)
        assert not math.isnan(result)

    def test_window_equals_sequence_length(self) -> None:
        """When window_size == T, should use all tokens. Must not NaN."""
        t = torch.randn(16, 64)
        result = participation_ratio(t, window_size=16)
        assert not math.isnan(result)

    def test_near_zero_tensor(self) -> None:
        """Very small but nonzero values. Should either compute or NaN, never crash."""
        t = torch.randn(16, 64) * 1e-15
        result = participation_ratio(t, window_size=16)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Test Class 3: R_V (the ratio) end-to-end
# ---------------------------------------------------------------------------

class TestComputeRV:
    """Verify R_V = PR(late) / PR(early) with synthetic V matrices."""

    def test_rv_identical_matrices(self) -> None:
        """If V_early == V_late, R_V must be exactly 1.0."""
        t = _make_tensor_with_known_svd([5.0, 3.0, 2.0, 1.0], seq_len=16, dim=64)
        rv = compute_rv(t, t, window=16)
        assert rv == pytest.approx(1.0, abs=1e-10)

    def test_rv_contraction(self) -> None:
        """Late matrix with fewer effective dimensions => R_V < 1.0.

        V_early has 4 equal SVs (PR=4), V_late has 1 dominant (PR~1).
        R_V = ~1/4 = 0.25.
        """
        v_early = _make_tensor_with_known_svd([1.0, 1.0, 1.0, 1.0], seq_len=16, dim=64)
        v_late = _make_tensor_with_known_svd([10.0, 0.01, 0.01, 0.01], seq_len=16, dim=64)

        rv = compute_rv(v_early, v_late, window=16)
        assert rv < 1.0, f"Expected R_V < 1.0 for contraction, got {rv}"

        # PR_early ~ 4.0, PR_late ~ 1.0
        pr_early = participation_ratio(v_early, window_size=16)
        pr_late = participation_ratio(v_late, window_size=16)
        assert pr_early == pytest.approx(4.0, abs=1e-4)
        assert pr_late == pytest.approx(1.0, abs=0.05)  # not exactly 1 due to small SVs

    def test_rv_expansion(self) -> None:
        """Late matrix with more effective dimensions => R_V > 1.0."""
        v_early = _make_tensor_with_known_svd([10.0, 0.01], seq_len=16, dim=64)
        v_late = _make_tensor_with_known_svd([1.0, 1.0], seq_len=16, dim=64)

        rv = compute_rv(v_early, v_late, window=16)
        assert rv > 1.0, f"Expected R_V > 1.0 for expansion, got {rv}"

    def test_rv_with_components_consistency(self) -> None:
        """compute_rv_with_components must return (rv, pr_early, pr_late)
        where rv == pr_late / pr_early.
        """
        v_early = _make_tensor_with_known_svd([4.0, 2.0, 1.0], seq_len=16, dim=64)
        v_late = _make_tensor_with_known_svd([3.0, 3.0, 3.0], seq_len=16, dim=64)

        rv, pr_e, pr_l = compute_rv_with_components(v_early, v_late, window=16)

        assert rv == pytest.approx(pr_l / pr_e, rel=1e-10), (
            f"R_V={rv} should equal PR_late/PR_early={pr_l / pr_e}"
        )

    def test_rv_none_early(self) -> None:
        """None V_early must return NaN."""
        v_late = torch.randn(16, 64)
        rv = compute_rv(None, v_late, window=16)
        assert math.isnan(rv)

    def test_rv_none_late(self) -> None:
        """None V_late must return NaN."""
        v_early = torch.randn(16, 64)
        rv = compute_rv(v_early, None, window=16)
        assert math.isnan(rv)

    def test_rv_both_none(self) -> None:
        """Both None must return NaN."""
        rv = compute_rv(None, None, window=16)
        assert math.isnan(rv)

    def test_rv_known_analytic_value(self) -> None:
        """End-to-end test with analytically known R_V.

        V_early: sigma = [2, 2] => PR = 2.0
        V_late:  sigma = [4, 2, 1] => PR = 441/273 ~ 1.6154
        R_V = 1.6154 / 2.0 = 0.80769...
        """
        v_early = _make_tensor_with_known_svd([2.0, 2.0], seq_len=16, dim=64)
        v_late = _make_tensor_with_known_svd([4.0, 2.0, 1.0], seq_len=16, dim=64)

        expected_pr_early = 2.0
        expected_pr_late = 441.0 / 273.0
        expected_rv = expected_pr_late / expected_pr_early

        rv, pr_e, pr_l = compute_rv_with_components(v_early, v_late, window=16)

        assert pr_e == pytest.approx(expected_pr_early, abs=1e-4)
        assert pr_l == pytest.approx(expected_pr_late, abs=1e-4)
        assert rv == pytest.approx(expected_rv, abs=1e-4)


# ---------------------------------------------------------------------------
# Test Class 4: Identity matrix special case
# ---------------------------------------------------------------------------

class TestPRIdentityMatrix:
    """PR(I_k) = k is a fundamental property. Test it for several k values."""

    @pytest.mark.parametrize("k", [1, 2, 4, 8, 12, 16])
    def test_pr_of_k_equal_singular_values_is_k(self, k: int) -> None:
        """PR of a matrix with k equal singular values must equal k.

        This is the defining property of participation ratio: it counts the
        number of effectively participating dimensions.
        """
        sigmas = [1.0] * k
        tensor = _make_tensor_with_known_svd(sigmas, seq_len=16, dim=64)
        pr = participation_ratio(tensor, window_size=16)
        assert pr == pytest.approx(float(k), abs=1e-5), (
            f"PR should be {k} for {k} equal SVs, got {pr}"
        )


# ---------------------------------------------------------------------------
# Test Class 5: Cross-implementation agreement (legacy vs canonical)
# ---------------------------------------------------------------------------

class TestLegacyAgreement:
    """Verify that geometric_lens/metrics.py and src/metrics/rv.py compute
    the same participation_ratio on identical inputs.

    The legacy implementation differs only in not calling .cpu() before SVD
    (irrelevant on CPU) and logging. The core formula is identical.

    We extract ONLY the participation_ratio function from the legacy file
    to avoid importing its model-dependent dependencies (transformers, hooks).
    """

    @pytest.fixture
    def legacy_pr(self) -> Callable[..., float] | None:
        """Extract the legacy participation_ratio by compiling its source.

        This avoids triggering the module-level import of transformers and
        src.core.hooks, which are not needed for the pure-math PR function.
        """
        legacy_path = PROJECT_ROOT / "src" / "metrics" / "rv.py"
        if not legacy_path.exists():
            pytest.skip("Legacy src/metrics/rv.py not found")
            return None

        import importlib.util
        import types

        # Create a minimal module that stubs out the problematic imports
        source = legacy_path.read_text()

        # Build a module with stubbed dependencies
        stub_module = types.ModuleType("_legacy_rv_stub")
        stub_module.__dict__["__name__"] = "_legacy_rv_stub"
        stub_module.__dict__["__file__"] = str(legacy_path)

        # Rewrite the source to remove the relative import and model-dependent code
        # We only need the participation_ratio function
        lines = source.split("\n")
        filtered: list[str] = []
        for line in lines:
            # Skip the relative import and transformers import
            if "from ..core.hooks" in line:
                continue
            if "from transformers" in line:
                continue
            filtered.append(line)

        clean_source = "\n".join(filtered)

        # Execute in a namespace that has the needed stdlib modules
        ns: dict = {
            "logging": __import__("logging"),
            "np": np,
            "torch": torch,
            "Optional": None,  # We need typing.Optional
        }
        # Add typing
        import typing
        ns["Optional"] = typing.Optional

        exec(compile(clean_source, str(legacy_path), "exec"), ns)

        fn = ns.get("participation_ratio")
        if fn is None:
            pytest.skip("Could not extract participation_ratio from legacy source")
            return None
        return fn

    def test_agreement_identity(
        self, legacy_pr: Callable[..., float], identity_16x64: torch.Tensor
    ) -> None:
        """Both implementations should return 16.0 for 16 equal singular values."""
        canonical = participation_ratio(identity_16x64, window_size=16)
        legacy = legacy_pr(identity_16x64, window_size=16)
        assert canonical == pytest.approx(legacy, rel=1e-10)

    def test_agreement_rank1(
        self, legacy_pr: Callable[..., float], rank1_16x64: torch.Tensor
    ) -> None:
        """Both implementations should return 1.0 for rank-1 matrix."""
        canonical = participation_ratio(rank1_16x64, window_size=16)
        legacy = legacy_pr(rank1_16x64, window_size=16)
        assert canonical == pytest.approx(legacy, rel=1e-10)

    def test_agreement_random_tensors(
        self, legacy_pr: Callable[..., float]
    ) -> None:
        """Both implementations should agree on 20 random tensors."""
        rng = torch.Generator().manual_seed(999)
        for _ in range(20):
            t = torch.randn(16, 64, generator=rng)
            canonical = participation_ratio(t, window_size=16)
            legacy = legacy_pr(t, window_size=16)
            assert canonical == pytest.approx(legacy, rel=1e-8), (
                f"Disagreement: canonical={canonical}, legacy={legacy}"
            )

    def test_agreement_none(self, legacy_pr: Callable[..., float]) -> None:
        """Both should return NaN for None."""
        canonical = participation_ratio(None)
        legacy = legacy_pr(None)
        assert math.isnan(canonical)
        assert math.isnan(legacy)

    def test_agreement_short_sequence(
        self, legacy_pr: Callable[..., float]
    ) -> None:
        """Both should return NaN for sequences shorter than window_size."""
        short = torch.randn(4, 64)
        canonical = participation_ratio(short, window_size=16)
        legacy = legacy_pr(short, window_size=16)
        assert math.isnan(canonical)
        assert math.isnan(legacy)


# ---------------------------------------------------------------------------
# Test Class 6: Random matrix sanity (statistical)
# ---------------------------------------------------------------------------

class TestRandomMatrixSanity:
    """Statistical tests on random matrices to catch gross formula errors."""

    def test_random_matrix_pr_near_min_dim(self) -> None:
        """For a random Gaussian matrix of shape (W, D) with W < D,
        PR should be close to W (all W directions used roughly equally).

        This is a well-known result: i.i.d. Gaussian matrices have
        approximately uniform singular value spectrum as W/D -> 0.
        """
        torch.manual_seed(42)
        # W=16, D=256 so W/D ~ 0.06 => spectrum fairly flat => PR ~ W
        t = torch.randn(16, 256)
        pr = participation_ratio(t, window_size=16)
        # Should be within 50% of W=16 for truly random data
        assert 8.0 < pr < 16.5, (
            f"Random Gaussian (16, 256) PR={pr}, expected near 16"
        )

    def test_pr_decreases_with_rank(self) -> None:
        """As the effective rank of the matrix decreases, PR must decrease."""
        prs: list[float] = []
        for k in [16, 8, 4, 2, 1]:
            sigmas = [1.0] * k
            tensor = _make_tensor_with_known_svd(sigmas, seq_len=16, dim=64)
            prs.append(participation_ratio(tensor, window_size=16))

        # Each PR should be strictly less than or equal to the previous
        for i in range(1, len(prs)):
            assert prs[i] <= prs[i - 1] + 1e-6, (
                f"PR should decrease with rank: {prs}"
            )


# ---------------------------------------------------------------------------
# Test Class 7: Numerical stability
# ---------------------------------------------------------------------------

class TestNumericalStability:
    """Verify the code handles extreme but valid numeric ranges."""

    def test_very_large_activations(self) -> None:
        """Large activations (e.g. pre-layernorm) should not overflow.

        The code casts to float64 before SVD, so 1e4 * normal should be fine.
        """
        t = torch.randn(16, 64) * 1e4
        result = participation_ratio(t, window_size=16)
        assert isinstance(result, float)
        assert not math.isnan(result)

    def test_very_small_activations(self) -> None:
        """Tiny activations should either compute or NaN, never raise."""
        t = torch.randn(16, 64) * 1e-8
        result = participation_ratio(t, window_size=16)
        assert isinstance(result, float)
        # May be NaN due to total_variance < 1e-10 guard, that's acceptable

    def test_mixed_scale_singular_values(self) -> None:
        """Singular values spanning orders of magnitude."""
        sigmas = [1e4, 1e2, 1.0, 1e-2]
        tensor = _make_tensor_with_known_svd(sigmas, seq_len=16, dim=64)
        pr = participation_ratio(tensor, window_size=16)
        expected = _pr_reference(np.array(sigmas))
        assert pr == pytest.approx(expected, rel=1e-3)

    def test_all_dtypes(self) -> None:
        """Every common dtype should produce a float result without raising."""
        base = torch.randn(16, 64)
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            t = base.to(dtype)
            result = participation_ratio(t, window_size=16)
            assert isinstance(result, float), f"Failed for dtype={dtype}"
