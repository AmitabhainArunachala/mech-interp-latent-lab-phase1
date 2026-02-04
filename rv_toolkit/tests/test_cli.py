"""
Tests for the rv_toolkit CLI.
"""

import subprocess
import sys
import pytest


class TestCLIHelp:
    """Test CLI help and basic commands."""
    
    def test_help(self):
        """CLI shows help without error."""
        result = subprocess.run(
            [sys.executable, "-m", "rv_toolkit.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "R_V metrics" in result.stdout
    
    def test_version(self):
        """CLI shows version."""
        result = subprocess.run(
            [sys.executable, "-m", "rv_toolkit.cli", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "rv_toolkit" in result.stdout


class TestCLIDemo:
    """Test CLI demo command."""
    
    def test_demo_runs(self):
        """Demo command runs without error."""
        result = subprocess.run(
            [sys.executable, "-m", "rv_toolkit.cli", "demo", "--n-samples", "5"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "R_V Toolkit Demonstration" in result.stdout
    
    def test_demo_shows_contraction(self):
        """Demo shows geometric contraction."""
        result = subprocess.run(
            [sys.executable, "-m", "rv_toolkit.cli", "demo", "--n-samples", "10"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Should show negative contraction
        assert "contraction" in result.stdout.lower()
    
    def test_demo_shows_effect_size(self):
        """Demo shows Cohen's d."""
        result = subprocess.run(
            [sys.executable, "-m", "rv_toolkit.cli", "demo", "--n-samples", "10"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Cohen's d" in result.stdout


class TestCLIPrompts:
    """Test CLI prompts command."""
    
    def test_prompts_list(self):
        """Prompts command lists prompts."""
        result = subprocess.run(
            [sys.executable, "-m", "rv_toolkit.cli", "prompts", "--count", "3"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "BASELINE" in result.stdout
        assert "RECURSIVE" in result.stdout
    
    def test_prompts_shows_count(self):
        """Prompts command shows total count."""
        result = subprocess.run(
            [sys.executable, "-m", "rv_toolkit.cli", "prompts", "--count", "2"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Total pairs available" in result.stdout


class TestCLICompute:
    """Test CLI compute command (error cases - requires tensor file)."""
    
    def test_compute_missing_file(self):
        """Compute command fails gracefully on missing file."""
        result = subprocess.run(
            [sys.executable, "-m", "rv_toolkit.cli", "compute", "nonexistent.pt"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "File not found" in result.stderr or "Error" in result.stderr


class TestCLIAnalyze:
    """Test CLI analyze command (error cases - requires results file)."""
    
    def test_analyze_missing_file(self):
        """Analyze command fails gracefully on missing file."""
        result = subprocess.run(
            [sys.executable, "-m", "rv_toolkit.cli", "analyze", "nonexistent.json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "File not found" in result.stderr or "Error" in result.stderr
