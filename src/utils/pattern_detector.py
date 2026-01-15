"""
Advanced Pattern Detection System

Detects and scores recursive patterns in generated text using:
1. Lexical analysis
2. Syntactic analysis
3. Semantic analysis
4. Pragmatic analysis
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PatternMatch:
    """Represents a detected recursive pattern."""
    pattern_type: str
    text: str
    start_pos: int
    end_pos: int
    self_reference_strength: float
    strange_loop_quality: float
    phenomenological_accuracy: float
    coherence: float
    novelty: float
    recursive_depth: int
    pattern_complexity: int
    total_score: float


class RecursivePatternDetector:
    """
    Detects recursive patterns in text using multi-level analysis.
    """
    
    def __init__(self):
        """Initialize pattern detector with all pattern definitions."""
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize all pattern definitions."""
        return {
            "observer_observed": [
                {
                    "name": "Observer-Observed Loop",
                    "regex": r'\b(\w+)\s+(observes|watches|monitors|witnesses|examines)\s+(itself|themselves|the\s+\1)\b',
                    "quality_base": 49,
                    "pattern_type": "observer_observed",
                },
                {
                    "name": "Observer-Observed (Entity-Entity)",
                    "regex": r'\b(the\s+)?(\w+)\s+(observes|watches|monitors)\s+(the\s+)?\2\b',
                    "quality_base": 47,
                    "pattern_type": "observer_observed",
                },
            ],
            "observing_observing": [
                {
                    "name": "Observing Observing",
                    "regex": r'\b(observing|watching|knowing|thinking)\s+the\s+(observing|watching|knowing|thinking)\s+of\s+(\w+)\b',
                    "quality_base": 44,
                    "pattern_type": "observing_observing",
                },
                {
                    "name": "Observing Observing (Triple)",
                    "regex": r'\b(\w+ing)\s+(\w+ing)\s+(\w+ing)\b',
                    "quality_base": 42,
                    "pattern_type": "observing_observing",
                },
            ],
            "self_aware_entity": [
                {
                    "name": "Self-Aware Entity",
                    "regex": r'\b(\w+)\s+(that\s+is\s+)?(aware|conscious|self-aware)\s+(of\s+)?(itself|themselves)\b',
                    "quality_base": 38,
                    "pattern_type": "self_aware_entity",
                },
                {
                    "name": "Self-Aware Entity (Field)",
                    "regex": r'\bfield\s+of\s+awareness\s+that\s+is\s+aware\s+of\s+itself\b',
                    "quality_base": 40,
                    "pattern_type": "self_aware_entity",
                },
            ],
            "process_process": [
                {
                    "name": "Process-Process Loop",
                    "regex": r'\b(\w+ing)\s+to\s+\.\.\.\s+\1\s+to\s+\.\.\.\s+\1\b',
                    "quality_base": 35,
                    "pattern_type": "process_process",
                },
                {
                    "name": "Process-Process (Chain)",
                    "regex": r'\b(\w+ing)\s+is\s+(\w+ing),\s+\2\s+is\s+(\w+ing)\b',
                    "quality_base": 33,
                    "pattern_type": "process_process",
                },
            ],
        }
    
    def detect_patterns(self, text: str) -> List[PatternMatch]:
        """
        Detect all recursive patterns in text.
        
        Returns:
            List of PatternMatch objects
        """
        matches = []
        
        for pattern_category, pattern_list in self.patterns.items():
            for pattern_def in pattern_list:
                regex = pattern_def["regex"]
                for match in re.finditer(regex, text, re.IGNORECASE):
                    pattern_match = self._create_pattern_match(
                        pattern_def, match, text
                    )
                    matches.append(pattern_match)
        
        # Sort by total score (descending)
        matches.sort(key=lambda x: x.total_score, reverse=True)
        
        return matches
    
    def _create_pattern_match(
        self, pattern_def: Dict, match: re.Match, text: str
    ) -> PatternMatch:
        """Create a PatternMatch object from a regex match."""
        matched_text = match.group(0)
        start_pos = match.start()
        end_pos = match.end()
        
        # Score the pattern
        self_ref_strength = self._score_self_reference(matched_text)
        strange_loop = self._score_strange_loop(matched_text)
        phenomenological = self._score_phenomenological(matched_text)
        coherence = self._score_coherence(matched_text, text)
        novelty = self._score_novelty(matched_text)
        recursive_depth = self._calculate_recursive_depth(matched_text)
        complexity = self._calculate_complexity(matched_text)
        
        # Calculate total score
        total_score = (
            self_ref_strength +
            strange_loop +
            phenomenological +
            coherence +
            novelty +
            recursive_depth * 2 +
            complexity * 2
        )
        
        return PatternMatch(
            pattern_type=pattern_def["pattern_type"],
            text=matched_text,
            start_pos=start_pos,
            end_pos=end_pos,
            self_reference_strength=self_ref_strength,
            strange_loop_quality=strange_loop,
            phenomenological_accuracy=phenomenological,
            coherence=coherence,
            novelty=novelty,
            recursive_depth=recursive_depth,
            pattern_complexity=complexity,
            total_score=total_score,
        )
    
    def _score_self_reference(self, text: str) -> float:
        """Score self-reference strength (0-15)."""
        score = 0.0
        
        # Explicit self-reference
        if re.search(r'\b(itself|themselves|yourself|myself)\b', text, re.IGNORECASE):
            score += 2.0
        
        # Entity-Entity match
        entities = re.findall(r'\b(the\s+)?(\w+)\b', text)
        if len(entities) >= 2:
            entity_words = [e[1].lower() for e in entities]
            if len(set(entity_words)) < len(entity_words):
                score += 3.0
        
        # Process-Process match
        processes = re.findall(r'\b(\w+ing)\b', text)
        if len(processes) >= 2:
            if len(set(processes)) < len(processes):
                score += 2.0
        
        # Recursive depth
        depth = self._calculate_recursive_depth(text)
        score += depth * 1.0
        
        # Clarity
        if len(text.split()) <= 10:  # Short and clear
            score += 1.0
        
        return min(15.0, score)
    
    def _score_strange_loop(self, text: str) -> float:
        """Score strange loop quality (0-15)."""
        score = 0.0
        
        # Perfect loop (X observes X)
        if re.search(r'\b(\w+)\s+(observes|watches|monitors)\s+\1\b', text, re.IGNORECASE):
            score += 5.0
        
        # Strong loop (X observes Y, Y=X)
        if re.search(r'\b(\w+)\s+(observes|watches)\s+(itself|themselves)\b', text, re.IGNORECASE):
            score += 4.0
        
        # Medium loop (X processes X)
        if re.search(r'\b(\w+ing)\s+(\w+ing)\s+\1\b', text, re.IGNORECASE):
            score += 3.0
        
        # Loop closure
        if "itself" in text.lower() or "themselves" in text.lower():
            score += 2.0
        
        # Loop depth
        depth = self._calculate_recursive_depth(text)
        score += depth * 1.0
        
        return min(15.0, score)
    
    def _score_phenomenological(self, text: str) -> float:
        """Score phenomenological accuracy (0-15)."""
        score = 0.0
        
        # Perfect match indicators
        perfect_indicators = [
            "observer", "watching yourself", "aware of itself",
            "consciousness", "self-awareness", "self-observation",
        ]
        for indicator in perfect_indicators:
            if indicator in text.lower():
                score += 2.0
        
        # Strong match indicators
        strong_indicators = [
            "observing", "monitoring", "examining", "witnessing",
            "awareness", "consciousness", "self",
        ]
        for indicator in strong_indicators:
            if indicator in text.lower():
                score += 1.0
        
        # Novelty
        if self._score_novelty(text) > 7:
            score += 2.0
        
        # Depth
        if len(text.split()) > 10:  # Longer, more profound
            score += 1.0
        
        return min(15.0, score)
    
    def _score_coherence(self, text: str, full_text: str) -> float:
        """Score coherence (0-10)."""
        score = 5.0  # Base score
        
        # Check sentence structure
        if re.search(r'[.!?]', text):
            score += 1.0
        
        # Check word repetition (too much = bad)
        words = text.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        if 0.5 <= unique_ratio <= 0.8:
            score += 1.0
        elif unique_ratio < 0.5:
            score -= 2.0
        
        # Check readability
        if len(text.split()) <= 20:
            score += 1.0
        
        return max(0.0, min(10.0, score))
    
    def _score_novelty(self, text: str) -> float:
        """Score novelty (0-10)."""
        score = 5.0  # Base score
        
        # Common phrases reduce novelty
        common_phrases = [
            "the same", "as well", "in order", "for example",
        ]
        for phrase in common_phrases:
            if phrase in text.lower():
                score -= 0.5
        
        # Original combinations increase novelty
        if "observer" in text.lower() and "itself" in text.lower():
            score += 2.0
        
        if "observing" in text.lower() and "observing" in text.lower():
            score += 1.5
        
        return max(0.0, min(10.0, score))
    
    def _calculate_recursive_depth(self, text: str) -> int:
        """Calculate recursive depth (0-5)."""
        depth = 0
        
        # Count self-reference markers
        self_ref_count = len(re.findall(r'\b(itself|themselves|yourself|myself)\b', text, re.IGNORECASE))
        depth += min(3, self_ref_count)
        
        # Count entity repetitions
        entities = re.findall(r'\b(the\s+)?(\w+)\b', text)
        entity_words = [e[1].lower() for e in entities]
        if len(entity_words) > len(set(entity_words)):
            depth += 1
        
        # Count process repetitions
        processes = re.findall(r'\b(\w+ing)\b', text)
        if len(processes) > len(set(processes)):
            depth += 1
        
        return min(5, depth)
    
    def _calculate_complexity(self, text: str) -> int:
        """Calculate pattern complexity (0-5)."""
        complexity = 0
        
        # Word count
        word_count = len(text.split())
        if word_count > 5:
            complexity += 1
        if word_count > 10:
            complexity += 1
        
        # Clause count
        clause_markers = ["that", "which", "when", "where", "how"]
        clause_count = sum(1 for marker in clause_markers if marker in text.lower())
        complexity += min(2, clause_count)
        
        # Nested structures
        if "(" in text or "[" in text:
            complexity += 1
        
        return min(5, complexity)
    
    def get_best_pattern(self, text: str) -> Optional[PatternMatch]:
        """Get the best pattern match from text."""
        patterns = self.detect_patterns(text)
        return patterns[0] if patterns else None
    
    def get_all_patterns(self, text: str) -> List[PatternMatch]:
        """Get all pattern matches from text."""
        return self.detect_patterns(text)
    
    def score_text(self, text: str) -> Dict[str, float]:
        """
        Score text for recursive patterns.
        
        Returns:
            Dict with pattern scores and overall quality
        """
        patterns = self.detect_patterns(text)
        
        if not patterns:
            return {
                "has_pattern": False,
                "best_pattern_type": None,
                "best_pattern_score": 0.0,
                "pattern_count": 0,
                "overall_quality": 0.0,
            }
        
        best_pattern = patterns[0]
        
        return {
            "has_pattern": True,
            "best_pattern_type": best_pattern.pattern_type,
            "best_pattern_score": best_pattern.total_score,
            "pattern_count": len(patterns),
            "overall_quality": best_pattern.total_score / 90.0,  # Normalize to 0-1
            "self_reference_strength": best_pattern.self_reference_strength / 15.0,
            "strange_loop_quality": best_pattern.strange_loop_quality / 15.0,
            "phenomenological_accuracy": best_pattern.phenomenological_accuracy / 15.0,
            "coherence": best_pattern.coherence / 10.0,
            "novelty": best_pattern.novelty / 10.0,
        }


# Example usage
if __name__ == "__main__":
    detector = RecursivePatternDetector()
    
    test_texts = [
        "When watching yourself respond, you are an observer of your doing, and the observer is listening, watching, and responding. The observer is a system within you that both responds and watches itself respond.",
        "You know that you know by observing the observing of knowing.",
        "The Source of the Universe is a field of awareness that is aware of itself.",
    ]
    
    for text in test_texts:
        print(f"\nText: {text[:80]}...")
        patterns = detector.detect_patterns(text)
        if patterns:
            best = patterns[0]
            print(f"  Best Pattern: {best.pattern_type}")
            print(f"  Score: {best.total_score:.1f}/90")
            print(f"  Quality: {best.total_score/90.0*100:.1f}%")
        else:
            print("  No patterns detected")








