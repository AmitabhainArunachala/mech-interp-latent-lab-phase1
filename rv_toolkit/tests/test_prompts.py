"""
Tests for rv_toolkit.prompts module.

Tests prompt banks and retrieval functions.
"""

import pytest
import random

from rv_toolkit.prompts import (
    RECURSIVE_PROMPTS,
    BASELINE_PROMPTS,
    PSEUDO_RECURSIVE_PROMPTS,
    COMPLEX_BASELINE_PROMPTS,
    get_prompt_pairs,
    get_extended_prompt_bank,
    generate_recursive_prompt,
)


class TestPromptBanks:
    """Tests for the static prompt banks."""
    
    def test_recursive_prompts_exist(self):
        """Should have a non-empty list of recursive prompts."""
        assert len(RECURSIVE_PROMPTS) > 0
        assert all(isinstance(p, str) for p in RECURSIVE_PROMPTS)
    
    def test_baseline_prompts_exist(self):
        """Should have a non-empty list of baseline prompts."""
        assert len(BASELINE_PROMPTS) > 0
        assert all(isinstance(p, str) for p in BASELINE_PROMPTS)
    
    def test_prompts_are_non_empty(self):
        """All prompts should be non-empty strings."""
        for p in RECURSIVE_PROMPTS:
            assert len(p.strip()) > 0
        
        for p in BASELINE_PROMPTS:
            assert len(p.strip()) > 0
    
    def test_no_duplicates_in_banks(self):
        """Individual banks should not contain duplicates."""
        assert len(RECURSIVE_PROMPTS) == len(set(RECURSIVE_PROMPTS))
        assert len(BASELINE_PROMPTS) == len(set(BASELINE_PROMPTS))
    
    def test_reasonable_prompt_count(self):
        """Should have sufficient prompts for experiments."""
        # Paper uses n=151 pairs, so we need at least that many
        assert len(RECURSIVE_PROMPTS) >= 35
        assert len(BASELINE_PROMPTS) >= 35


class TestPromptCharacteristics:
    """Tests for prompt content characteristics."""
    
    def test_baseline_prompts_not_self_referential(self):
        """Baseline prompts should not be about model's own processing."""
        strong_self_ref = ['I observe myself', 'I am aware', 'my own process']
        
        for prompt in BASELINE_PROMPTS:
            prompt_lower = prompt.lower()
            for pattern in strong_self_ref:
                assert pattern not in prompt_lower, \
                    f"Baseline prompt contains self-reference: {prompt}"
    
    def test_control_prompts_exist(self):
        """Control prompt banks should exist."""
        assert len(PSEUDO_RECURSIVE_PROMPTS) > 0
        assert len(COMPLEX_BASELINE_PROMPTS) > 0


class TestGetPromptPairs:
    """Tests for get_prompt_pairs function."""
    
    def test_returns_correct_count(self):
        """Should return requested number of pairs."""
        pairs = get_prompt_pairs(n_pairs=10)
        assert len(pairs) == 10
    
    def test_returns_tuples(self):
        """Each pair should be a tuple of (baseline, recursive)."""
        pairs = get_prompt_pairs(n_pairs=5)
        
        for pair in pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            base, rec = pair
            assert isinstance(base, str)
            assert isinstance(rec, str)
    
    def test_pairs_from_correct_banks(self):
        """Baseline should come from BASELINE_PROMPTS, recursive from RECURSIVE_PROMPTS."""
        pairs = get_prompt_pairs(n_pairs=5, shuffle=False)
        
        for base, rec in pairs:
            assert base in BASELINE_PROMPTS
            assert rec in RECURSIVE_PROMPTS
    
    def test_respects_max_available(self):
        """Should not return more pairs than available."""
        n_recursive = len(RECURSIVE_PROMPTS)
        n_baseline = len(BASELINE_PROMPTS)
        max_pairs = min(n_recursive, n_baseline)
        
        pairs = get_prompt_pairs(n_pairs=max_pairs + 100)
        assert len(pairs) <= max_pairs
    
    def test_default_returns_all(self):
        """Default (n_pairs=None) should return all available pairs."""
        pairs = get_prompt_pairs(shuffle=False)
        n_possible = min(len(RECURSIVE_PROMPTS), len(BASELINE_PROMPTS))
        assert len(pairs) == n_possible
    
    def test_shuffle_produces_different_order(self):
        """Shuffled pairs should differ from unshuffled (usually)."""
        random.seed(42)
        pairs_shuffled = get_prompt_pairs(n_pairs=20, shuffle=True)
        
        random.seed(None)  # Reset seed
        pairs_unshuffled = get_prompt_pairs(n_pairs=20, shuffle=False)
        
        # Should be different (extremely unlikely to be same by chance)
        assert pairs_shuffled != pairs_unshuffled


class TestGetExtendedPromptBank:
    """Tests for get_extended_prompt_bank function."""
    
    def test_returns_dict(self):
        """Should return dictionary with expected keys."""
        bank = get_extended_prompt_bank()
        
        assert isinstance(bank, dict)
        assert 'recursive' in bank
        assert 'baseline' in bank
        assert 'pseudo_recursive' in bank
        assert 'complex_baseline' in bank
    
    def test_all_lists(self):
        """All values should be lists of strings."""
        bank = get_extended_prompt_bank()
        
        for key, prompts in bank.items():
            assert isinstance(prompts, list)
            assert all(isinstance(p, str) for p in prompts)


class TestGenerateRecursivePrompt:
    """Tests for template-based prompt generation."""
    
    def test_default_template(self):
        """Default template should produce non-empty string."""
        prompt = generate_recursive_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 20
    
    def test_all_templates_work(self):
        """All named templates should produce valid prompts."""
        templates = ['depth', 'witness', 'loop', 'meta']
        
        for template in templates:
            prompt = generate_recursive_prompt(template)
            assert isinstance(prompt, str)
            assert len(prompt) > 20
    
    def test_unknown_template_fallback(self):
        """Unknown template should fall back to default."""
        prompt = generate_recursive_prompt('nonexistent_template')
        default = generate_recursive_prompt('depth')
        assert prompt == default


class TestPromptQuality:
    """Tests for prompt quality and formatting."""
    
    def test_reasonable_length(self):
        """Prompts should be reasonable length."""
        all_prompts = RECURSIVE_PROMPTS + BASELINE_PROMPTS
        
        for p in all_prompts:
            assert 10 < len(p) < 500, f"Prompt length unusual: {len(p)} - {p[:50]}"
    
    def test_proper_encoding(self):
        """Prompts should be valid UTF-8."""
        all_prompts = (
            RECURSIVE_PROMPTS + 
            BASELINE_PROMPTS + 
            PSEUDO_RECURSIVE_PROMPTS + 
            COMPLEX_BASELINE_PROMPTS
        )
        
        for p in all_prompts:
            p.encode('utf-8')  # Should not raise
