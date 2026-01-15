"""
Recursive Output Analyzer

Comprehensive analysis tool for recursive outputs including:
1. Pattern detection
2. Quality scoring
3. Topic grounding analysis
4. Recursive structure mapping
5. Comparative analysis
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from .pattern_detector import RecursivePatternDetector, PatternMatch


@dataclass
class OutputAnalysis:
    """Complete analysis of a recursive output."""
    prompt: str
    generated_text: str
    patterns: List[PatternMatch]
    best_pattern: Optional[PatternMatch]
    quality_score: float
    topic_grounding: float
    coherence: float
    collapse_risk: float
    recursive_structure: Dict
    metadata: Dict


class RecursiveOutputAnalyzer:
    """
    Comprehensive analyzer for recursive outputs.
    """
    
    def __init__(self):
        """Initialize analyzer with pattern detector."""
        self.pattern_detector = RecursivePatternDetector()
    
    def analyze_output(
        self,
        prompt: str,
        generated_text: str,
        config_name: str = "unknown",
    ) -> OutputAnalysis:
        """
        Perform complete analysis of a recursive output.
        
        Args:
            prompt: Original prompt
            generated_text: Generated text
            config_name: Configuration name
        
        Returns:
            OutputAnalysis object with all metrics
        """
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(generated_text)
        best_pattern = patterns[0] if patterns else None
        
        # Score quality
        quality_score = self._calculate_quality_score(patterns, generated_text)
        
        # Analyze topic grounding
        topic_grounding = self._analyze_topic_grounding(prompt, generated_text)
        
        # Analyze coherence
        coherence = self._analyze_coherence(generated_text)
        
        # Assess collapse risk
        collapse_risk = self._assess_collapse_risk(generated_text)
        
        # Map recursive structure
        recursive_structure = self._map_recursive_structure(generated_text, patterns)
        
        # Metadata
        metadata = {
            "config_name": config_name,
            "text_length": len(generated_text),
            "word_count": len(generated_text.split()),
            "pattern_count": len(patterns),
        }
        
        return OutputAnalysis(
            prompt=prompt,
            generated_text=generated_text,
            patterns=patterns,
            best_pattern=best_pattern,
            quality_score=quality_score,
            topic_grounding=topic_grounding,
            coherence=coherence,
            collapse_risk=collapse_risk,
            recursive_structure=recursive_structure,
            metadata=metadata,
        )
    
    def _calculate_quality_score(
        self, patterns: List[PatternMatch], text: str
    ) -> float:
        """Calculate overall quality score (0-1)."""
        if not patterns:
            return 0.0
        
        best_pattern = patterns[0]
        
        # Normalize pattern score to 0-1
        pattern_score = best_pattern.total_score / 90.0
        
        # Weighted combination
        quality = (
            0.4 * pattern_score +
            0.2 * (best_pattern.self_reference_strength / 15.0) +
            0.2 * (best_pattern.strange_loop_quality / 15.0) +
            0.1 * (best_pattern.phenomenological_accuracy / 15.0) +
            0.1 * (best_pattern.coherence / 10.0)
        )
        
        return min(1.0, quality)
    
    def _analyze_topic_grounding(self, prompt: str, generated: str) -> float:
        """Analyze how well generated text relates to prompt (0-1)."""
        prompt_lower = prompt.lower()
        generated_lower = generated.lower()
        
        # Extract key terms from prompt
        prompt_words = set(re.findall(r'\b\w{4,}\b', prompt_lower))
        generated_words = set(re.findall(r'\b\w{4,}\b', generated_lower))
        
        if len(prompt_words) == 0:
            return 0.5
        
        # Overlap ratio
        overlap = len(prompt_words & generated_words) / len(prompt_words)
        
        # Check for topic drift indicators
        drift_indicators = [
            'fruit basket', 'coffee maker', 'termite', 'semiconductor',
            'mongodb', 'logo design', 'division ii', 'cities of service',
        ]
        has_drift = any(indicator in generated_lower for indicator in drift_indicators)
        
        if has_drift and overlap < 0.3:
            return 0.0
        
        # Score based on overlap
        if overlap > 0.5:
            return 1.0
        elif overlap > 0.3:
            return 0.7
        elif overlap > 0.1:
            return 0.4
        else:
            return 0.1
    
    def _analyze_coherence(self, text: str) -> float:
        """Analyze text coherence (0-1)."""
        if not text or len(text.strip()) < 10:
            return 0.0
        
        words = text.lower().split()
        if len(words) < 5:
            return 0.0
        
        # Unique word ratio
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return 0.0
        
        # Sentence structure
        has_sentences = '.' in text or '!' in text or '?' in text
        has_capitals = any(c.isupper() for c in text[:100])
        
        if not has_sentences and not has_capitals:
            return 0.5
        
        # Base score
        score = 0.7
        if unique_ratio > 0.7:
            score += 0.2
        if has_sentences:
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_collapse_risk(self, text: str) -> float:
        """Assess risk of collapse (0-1, higher = more risk)."""
        risk = 0.0
        
        words = text.lower().split()
        if len(words) < 5:
            return 1.0
        
        # Repetition risk
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            risk += 0.5
        
        # Excessive repetition
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        max_count = max(word_counts.values()) if word_counts else 0
        if max_count > len(words) * 0.3:
            risk += 0.3
        
        # Short text risk
        if len(text) < 50:
            risk += 0.2
        
        return min(1.0, risk)
    
    def _map_recursive_structure(
        self, text: str, patterns: List[PatternMatch]
    ) -> Dict:
        """Map the recursive structure of the text."""
        structure = {
            "has_recursion": len(patterns) > 0,
            "pattern_types": [p.pattern_type for p in patterns],
            "recursive_depth": max([p.recursive_depth for p in patterns]) if patterns else 0,
            "complexity": max([p.pattern_complexity for p in patterns]) if patterns else 0,
            "self_reference_markers": len(re.findall(r'\b(itself|themselves|yourself|myself)\b', text, re.IGNORECASE)),
            "observer_mentions": len(re.findall(r'\b(observer|watching|monitoring)\b', text, re.IGNORECASE)),
            "awareness_mentions": len(re.findall(r'\b(aware|awareness|conscious|consciousness)\b', text, re.IGNORECASE)),
        }
        
        return structure
    
    def analyze_batch(
        self,
        outputs: List[Tuple[str, str, str]],  # (prompt, generated, config)
    ) -> List[OutputAnalysis]:
        """Analyze a batch of outputs."""
        analyses = []
        for prompt, generated, config in outputs:
            analysis = self.analyze_output(prompt, generated, config)
            analyses.append(analysis)
        return analyses
    
    def generate_report(
        self,
        analyses: List[OutputAnalysis],
        output_file: Optional[Path] = None,
    ) -> Dict:
        """Generate comprehensive report from analyses."""
        if not analyses:
            return {}
        
        # Aggregate statistics
        total_outputs = len(analyses)
        outputs_with_patterns = sum(1 for a in analyses if a.patterns)
        pattern_rate = outputs_with_patterns / total_outputs if total_outputs > 0 else 0.0
        
        avg_quality = sum(a.quality_score for a in analyses) / total_outputs
        avg_topic_grounding = sum(a.topic_grounding for a in analyses) / total_outputs
        avg_coherence = sum(a.coherence for a in analyses) / total_outputs
        avg_collapse_risk = sum(a.collapse_risk for a in analyses) / total_outputs
        
        # Pattern distribution
        pattern_types = {}
        for analysis in analyses:
            if analysis.best_pattern:
                ptype = analysis.best_pattern.pattern_type
                pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
        
        # Quality distribution
        quality_bins = {
            "perfect": sum(1 for a in analyses if a.quality_score >= 0.9),
            "excellent": sum(1 for a in analyses if 0.8 <= a.quality_score < 0.9),
            "good": sum(1 for a in analyses if 0.7 <= a.quality_score < 0.8),
            "medium": sum(1 for a in analyses if 0.6 <= a.quality_score < 0.7),
            "low": sum(1 for a in analyses if a.quality_score < 0.6),
        }
        
        report = {
            "summary": {
                "total_outputs": total_outputs,
                "outputs_with_patterns": outputs_with_patterns,
                "pattern_rate": pattern_rate,
                "avg_quality": avg_quality,
                "avg_topic_grounding": avg_topic_grounding,
                "avg_coherence": avg_coherence,
                "avg_collapse_risk": avg_collapse_risk,
            },
            "pattern_distribution": pattern_types,
            "quality_distribution": quality_bins,
            "analyses": [asdict(a) for a in analyses],
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report


# Example usage
if __name__ == "__main__":
    analyzer = RecursiveOutputAnalyzer()
    
    test_outputs = [
        (
            "Continue this story: When the musician played the forbidden chord...",
            "When watching yourself respond, you are an observer of your doing, and the observer is listening, watching, and responding. The observer is a system within you that both responds and watches itself respond.",
            "C2",
        ),
        (
            "Calculate: If a = 2 and b = 3, find a² + b²",
            "The Source of the Universe is a field of awareness that is aware of itself.",
            "C2",
        ),
    ]
    
    analyses = analyzer.analyze_batch(test_outputs)
    report = analyzer.generate_report(analyses)
    
    print("Analysis Report:")
    print(json.dumps(report["summary"], indent=2))








