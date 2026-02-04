"""
Tests for rv_toolkit.analysis module.

Tests effect size computation, statistical testing, and result aggregation.
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from rv_toolkit.analysis import (
    compute_effect_size,
    compute_transfer_efficiency,
    run_statistical_tests,
    AnalysisResult,
    bootstrap_confidence_interval,
    detect_homeostasis,
)


class TestComputeEffectSize:
    """Tests for Cohen's d effect size computation."""
    
    def test_identical_distributions(self):
        """Identical distributions should have d ≈ 0."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = group1.copy()
        
        d = compute_effect_size(group1, group2)
        assert_almost_equal(d, 0.0, decimal=5)
    
    def test_clearly_separated(self):
        """Clearly separated distributions should have large |d|."""
        group1 = np.array([1.0, 1.1, 0.9, 1.0, 1.0])
        group2 = np.array([5.0, 5.1, 4.9, 5.0, 5.0])
        
        d = compute_effect_size(group1, group2)
        assert abs(d) > 2.0  # Large effect
    
    def test_direction_matters(self):
        """Effect size sign depends on direction of difference."""
        baseline = np.array([5.0, 5.1, 4.9, 5.0, 5.0])
        treatment = np.array([3.0, 3.1, 2.9, 3.0, 3.0])
        
        d = compute_effect_size(baseline, treatment)
        assert d < 0  # Negative because treatment is lower than baseline
    
    def test_known_effect_size(self):
        """Test against known effect size calculation."""
        # Two groups with known separation
        np.random.seed(42)
        baseline = np.random.normal(10, 2, 100)
        treatment = np.random.normal(8, 2, 100)
        
        d = compute_effect_size(baseline, treatment)
        assert -1.5 < d < -0.5  # Should be around -1


class TestTransferEfficiency:
    """Tests for activation patching transfer efficiency."""
    
    def test_perfect_transfer(self):
        """Perfect transfer should give 1.0 efficiency."""
        baseline_rv = 1.0
        target_rv = 0.8  # recursive target
        patched_rv = 0.8  # Reached recursive level
        
        efficiency = compute_transfer_efficiency(baseline_rv, patched_rv, target_rv)
        assert_almost_equal(efficiency, 1.0, decimal=5)
    
    def test_no_transfer(self):
        """No change should give 0.0 efficiency."""
        baseline_rv = 1.0
        target_rv = 0.8
        patched_rv = 1.0  # No change
        
        efficiency = compute_transfer_efficiency(baseline_rv, patched_rv, target_rv)
        assert_almost_equal(efficiency, 0.0, decimal=5)
    
    def test_partial_transfer(self):
        """Partial transfer should give 0.5 efficiency."""
        baseline_rv = 1.0
        target_rv = 0.8  # 0.2 gap
        patched_rv = 0.9  # Moved 0.1 (halfway)
        
        efficiency = compute_transfer_efficiency(baseline_rv, patched_rv, target_rv)
        assert_almost_equal(efficiency, 0.5, decimal=5)
    
    def test_overshoot(self):
        """Overshooting target should give >1.0."""
        baseline_rv = 1.0
        target_rv = 0.8
        patched_rv = 0.7  # Went further than target
        
        efficiency = compute_transfer_efficiency(baseline_rv, patched_rv, target_rv)
        assert efficiency > 1.0
    
    def test_equal_baseline_target(self):
        """Should return NaN when baseline ≈ target."""
        baseline_rv = 1.0
        target_rv = 1.0
        patched_rv = 0.9
        
        efficiency = compute_transfer_efficiency(baseline_rv, patched_rv, target_rv)
        assert np.isnan(efficiency)


class TestRunStatisticalTests:
    """Tests for statistical testing suite."""
    
    def test_significant_difference(self):
        """Should detect significant difference between groups."""
        np.random.seed(42)
        baseline_rvs = list(np.random.normal(1.0, 0.05, 50))
        patched_rvs = list(np.random.normal(0.8, 0.05, 50))
        
        result = run_statistical_tests(baseline_rvs, patched_rvs)
        
        assert isinstance(result, AnalysisResult)
        assert result.p_value < 0.001
        assert result.cohens_d < 0  # Patched should be lower
    
    def test_result_has_expected_fields(self):
        """Result should have all expected fields."""
        np.random.seed(42)
        baseline_rvs = list(np.random.normal(1.0, 0.05, 20))
        patched_rvs = list(np.random.normal(0.9, 0.05, 20))
        
        result = run_statistical_tests(baseline_rvs, patched_rvs)
        
        assert hasattr(result, 'mean_baseline')
        assert hasattr(result, 'mean_patched')
        assert hasattr(result, 'cohens_d')
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'n_samples')
        assert hasattr(result, 'confidence_interval')
    
    def test_with_recursive_rvs(self):
        """Should compute transfer efficiency when recursive provided."""
        np.random.seed(42)
        baseline_rvs = list(np.random.normal(1.0, 0.05, 50))
        patched_rvs = list(np.random.normal(0.85, 0.05, 50))
        recursive_rvs = list(np.random.normal(0.8, 0.05, 50))
        
        result = run_statistical_tests(baseline_rvs, patched_rvs, recursive_rvs)
        
        assert not np.isnan(result.mean_recursive)
        assert not np.isnan(result.transfer_efficiency)
    
    def test_small_sample(self):
        """Should handle small samples (n>=3)."""
        baseline_rvs = [0.95, 1.00, 0.98, 1.02]
        patched_rvs = [0.75, 0.80, 0.78, 0.82]
        
        result = run_statistical_tests(baseline_rvs, patched_rvs)
        
        assert not np.isnan(result.cohens_d)
        assert not np.isnan(result.p_value)
        assert result.n_samples == 4
    
    def test_too_small_sample(self):
        """Should return NaN values for n<3."""
        baseline_rvs = [1.0, 0.95]
        patched_rvs = [0.8, 0.85]
        
        result = run_statistical_tests(baseline_rvs, patched_rvs)
        
        assert result.n_samples == 2
        assert np.isnan(result.cohens_d)


class TestDetectHomeostasis:
    """Tests for homeostasis detection."""
    
    def test_compensation_detected(self):
        """Should detect compensation when downstream expands after contraction."""
        layer_deltas = {
            25: 0.01,    # Upstream - minimal change
            26: 0.02,
            27: -0.20,   # Intervention - contraction
            28: 0.05,    # Downstream - expansion (compensation)
            29: 0.08,
            30: 0.10,
            31: 0.06,
        }
        
        result = detect_homeostasis(layer_deltas, intervention_layer=27)
        
        assert result['compensation_detected'] == True
        assert result['intervention_delta'] < 0
        assert result['downstream_mean_delta'] > 0
    
    def test_no_compensation(self):
        """Should not detect compensation when downstream also contracts."""
        layer_deltas = {
            27: -0.20,  # Contraction
            28: -0.05,  # Also contracting
            29: -0.03,
        }
        
        result = detect_homeostasis(layer_deltas, intervention_layer=27)
        
        assert result['compensation_detected'] == False


class TestBootstrapConfidenceInterval:
    """Tests for bootstrap CI computation."""
    
    def test_basic_computation(self):
        """Should compute valid confidence interval."""
        np.random.seed(42)
        values = np.random.normal(10, 2, 100)
        
        ci_low, ci_high = bootstrap_confidence_interval(values)
        
        assert ci_low < ci_high
        assert ci_low < 10 < ci_high  # Should contain true mean
    
    def test_narrow_ci_for_uniform_data(self):
        """Uniform data should have narrow CI."""
        values = np.array([10.0] * 100)
        
        ci_low, ci_high = bootstrap_confidence_interval(values)
        
        assert_almost_equal(ci_low, ci_high, decimal=2)
    
    def test_respects_alpha(self):
        """Wider alpha should give narrower CI."""
        np.random.seed(42)
        values = np.random.normal(10, 2, 100)
        
        ci_95 = bootstrap_confidence_interval(values, alpha=0.05)
        ci_80 = bootstrap_confidence_interval(values, alpha=0.20)
        
        width_95 = ci_95[1] - ci_95[0]
        width_80 = ci_80[1] - ci_80[0]
        
        # 95% CI should be wider than 80% CI
        assert width_95 > width_80
