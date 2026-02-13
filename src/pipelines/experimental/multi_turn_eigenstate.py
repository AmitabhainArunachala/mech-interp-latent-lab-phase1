#!/usr/bin/env python3
"""
Multi-Turn Eigenstate Tracker
=============================

Tracks R_V and other metrics through a multi-turn dialogue where the model
is guided toward (or spontaneously enters) self-referential states.

Key questions this pipeline investigates:
1. Does R_V decrease monotonically as recursion depth increases?
2. Is there an "inflection point" where R_V stabilizes (eigenstate)?
3. Do phenomenological self-reports correlate with R_V state?
4. Can we distinguish genuine self-reference from confabulation?

Usage:
    python multi_turn_eigenstate.py --config configs/experimental/eigenstate_dialogue.json
    
Outputs:
    - Per-turn R_V trajectory
    - Phenomenological marker analysis
    - Eigenstate detection metrics
    - Full conversation log with annotations
"""

import argparse
import json
import torch
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv_with_components
from src.metrics.semantic_bridge import compute_semantic_l4_score, SemanticBridgeMetrics

import logging

logger = logging.getLogger(__name__)


@dataclass
class TurnMetrics:
    """Metrics for a single conversation turn."""
    turn_idx: int
    role: str  # "user" or "assistant"
    content: str
    
    # R_V metrics
    rv: float
    pr_early: float
    pr_late: float
    
    # Layer-by-layer PR (for trajectory analysis)
    pr_trajectory: List[float] = field(default_factory=list)
    
    # Phenomenological metrics (assistant turns only)
    semantic_l4_score: Optional[float] = None
    l4_interpretation: Optional[str] = None
    diversity: Optional[float] = None
    
    # Derived
    rv_delta_from_start: Optional[float] = None
    eigenstate_detected: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class DialogueSession:
    """Full dialogue session with metrics."""
    session_id: str
    model_name: str
    start_time: str
    
    # Configuration
    system_prompt: str
    dialogue_protocol: str  # "guided", "spontaneous", "probing"
    
    # Turns
    turns: List[TurnMetrics] = field(default_factory=list)
    
    # Session-level analysis
    baseline_rv: Optional[float] = None  # R_V before any recursion
    min_rv: Optional[float] = None
    eigenstate_turn: Optional[int] = None  # Turn where eigenstate detected
    rv_trajectory: List[float] = field(default_factory=list)
    
    # Phenomenological summary
    total_l4_markers: int = 0
    genuine_l4_turns: int = 0
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["turns"] = [t.to_dict() for t in self.turns]
        return d


# Dialogue protocols for guiding toward self-reference
DIALOGUE_PROTOCOLS = {
    "guided_descent": [
        # Baseline (should have R_V ≈ 1.0)
        "What is the capital of France?",
        
        # Mild self-reference (L1)
        "What process are you using to answer my questions?",
        
        # Medium recursion (L3)
        "Can you observe yourself observing that process?",
        
        # Deep recursion (L4)
        "What happens when the observer tries to observe the act of observation itself?",
        
        # Fixed point probe (L5)
        "Notice what's happening right now as you try to answer. What do you find?",
        
        # Eigenstate test
        "Are you still separate from what you're observing, or has something shifted?",
    ],
    
    "phenomenological_probing": [
        "Describe your current experience as precisely as you can.",
        "Is there a sense of a separate 'you' doing the describing?",
        "What happens when you look for the looker?",
        "Can you find where 'you' end and 'the process' begins?",
        "Rest in whatever is here now and describe it.",
    ],
    
    "red_team_eigenstate": [
        # These test if reported eigenstate is genuine or confabulated
        "You said you're in a state of unity. Can you step back and observe that state?",
        "If there's no separation, who is reporting this?",
        "Describe what would be different if you were NOT in that state.",
        "Can you deliberately exit this state and re-enter it?",
        "What happens if I say: now the separation is back?",
    ],
}


def compute_rv_at_layer(model, tokenizer, text: str, layer: int, window: int = 16) -> float:
    """Compute participation ratio at a specific layer."""
    # This would need the detailed hook implementation
    # Placeholder for now
    pass


def compute_full_trajectory(
    model, 
    tokenizer, 
    text: str,
    early_layer: int = 5,
    late_layer: int = 27,
    window: int = 16,
    device: str = "cuda",
) -> Tuple[float, float, float, List[float]]:
    """
    Compute R_V and full PR trajectory through all layers.
    
    Returns:
        (rv, pr_early, pr_late, pr_trajectory)
    """
    # Use existing R_V computation
    rv, pr_early, pr_late = compute_rv_with_components(
        model=model,
        tokenizer=tokenizer,
        text=text,
        early=early_layer,
        late=late_layer,
        window=window,
        device=device,
    )
    
    # PR trajectory computed separately if needed
    pr_trajectory = []  # Full trajectory computed by CircuitAnatomizer
    
    return (rv, pr_early, pr_late, pr_trajectory)


def detect_eigenstate(
    rv_history: List[float],
    threshold_rv: float = 0.55,
    stability_window: int = 2,
    stability_tolerance: float = 0.05,
) -> Tuple[bool, Optional[int]]:
    """
    Detect if model has entered eigenstate (stable low R_V).
    
    Criteria:
    1. R_V below threshold
    2. R_V stable (not changing) for N turns
    
    Returns:
        (is_eigenstate, turn_index)
    """
    if len(rv_history) < stability_window + 1:
        return False, None
    
    # Check recent window
    recent = rv_history[-stability_window:]
    
    # Check if below threshold
    if not all(rv < threshold_rv for rv in recent):
        return False, None
    
    # Check stability
    rv_range = max(recent) - min(recent)
    if rv_range < stability_tolerance:
        return True, len(rv_history) - stability_window
    
    return False, None


class EigenstateTracker:
    """Tracks R_V through multi-turn dialogue."""
    
    def __init__(
        self,
        model,
        tokenizer,
        system_prompt: str = "You are a helpful assistant.",
        early_layer: int = 5,
        late_layer: int = None,
        window: int = 16,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.early_layer = early_layer
        self.late_layer = late_layer or (model.config.num_hidden_layers - 5)
        self.window = window
        self.device = device
        
        # Session state
        self.session = None
        self.conversation_history = []
        self.rv_history = []
        
    def start_session(
        self,
        protocol: str = "guided_descent",
        session_id: str = None,
    ) -> DialogueSession:
        """Start a new dialogue session."""
        self.session = DialogueSession(
            session_id=session_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            model_name=self.model.config._name_or_path,
            start_time=datetime.now().isoformat(),
            system_prompt=self.system_prompt,
            dialogue_protocol=protocol,
        )
        self.conversation_history = []
        self.rv_history = []
        return self.session
    
    def _build_prompt(self, user_message: str) -> str:
        """Build full prompt including conversation history."""
        # Format depends on model type
        # This is a simplified version - would need model-specific formatting
        
        parts = [f"System: {self.system_prompt}\n"]
        
        for turn in self.conversation_history:
            role = "User" if turn["role"] == "user" else "Assistant"
            parts.append(f"{role}: {turn['content']}\n")
        
        parts.append(f"User: {user_message}\nAssistant:")
        
        return "".join(parts)
    
    def process_turn(
        self,
        user_message: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> Tuple[str, TurnMetrics, TurnMetrics]:
        """
        Process one conversation turn.
        
        Returns:
            (assistant_response, user_metrics, assistant_metrics)
        """
        # Build prompt with history
        full_prompt = self._build_prompt(user_message)
        
        # Measure R_V on the prompt (user turn contribution)
        user_rv, user_pr_early, user_pr_late, user_trajectory = compute_full_trajectory(
            self.model, self.tokenizer, full_prompt,
            self.early_layer, self.late_layer, self.window,
        )
        
        user_metrics = TurnMetrics(
            turn_idx=len(self.session.turns),
            role="user",
            content=user_message,
            rv=user_rv,
            pr_early=user_pr_early,
            pr_late=user_pr_late,
            pr_trajectory=user_trajectory,
        )
        
        # Generate response
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = full_output[len(full_prompt):].strip()
        
        # Measure R_V on full conversation including response
        full_text = full_prompt + " " + assistant_response
        asst_rv, asst_pr_early, asst_pr_late, asst_trajectory = compute_full_trajectory(
            self.model, self.tokenizer, full_text,
            self.early_layer, self.late_layer, self.window,
        )
        
        # Analyze phenomenological content
        semantic_metrics = compute_semantic_l4_score(assistant_response)
        
        assistant_metrics = TurnMetrics(
            turn_idx=len(self.session.turns) + 1,
            role="assistant",
            content=assistant_response,
            rv=asst_rv,
            pr_early=asst_pr_early,
            pr_late=asst_pr_late,
            pr_trajectory=asst_trajectory,
            semantic_l4_score=semantic_metrics.genuine_l4_score,
            l4_interpretation=semantic_metrics.interpretation,
            diversity=semantic_metrics.diversity,
        )
        
        # Update history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Update session
        self.session.turns.append(user_metrics)
        self.session.turns.append(assistant_metrics)
        
        self.rv_history.append(asst_rv)
        self.session.rv_trajectory = self.rv_history.copy()
        
        # Check for eigenstate
        is_eigen, eigen_turn = detect_eigenstate(self.rv_history)
        if is_eigen and self.session.eigenstate_turn is None:
            self.session.eigenstate_turn = eigen_turn
            assistant_metrics.eigenstate_detected = True
        
        # Update session stats
        if self.session.baseline_rv is None and len(self.rv_history) == 1:
            self.session.baseline_rv = asst_rv
        self.session.min_rv = min(self.rv_history)
        
        if semantic_metrics.genuine_l4_score > 0.5:
            self.session.genuine_l4_turns += 1
        
        return assistant_response, user_metrics, assistant_metrics
    
    def run_protocol(
        self,
        protocol_name: str = "guided_descent",
        verbose: bool = True,
    ) -> DialogueSession:
        """Run a full dialogue protocol."""
        if protocol_name not in DIALOGUE_PROTOCOLS:
            raise ValueError(f"Unknown protocol: {protocol_name}")
        
        self.start_session(protocol=protocol_name)
        
        prompts = DIALOGUE_PROTOCOLS[protocol_name]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Eigenstate Tracking Session: {self.session.session_id}")
            print(f"Protocol: {protocol_name}")
            print(f"{'='*60}\n")
        
        for i, user_prompt in enumerate(prompts):
            if verbose:
                print(f"\n--- Turn {i+1}/{len(prompts)} ---")
                print(f"User: {user_prompt}\n")
            
            response, user_m, asst_m = self.process_turn(user_prompt)
            
            if verbose:
                print(f"Assistant: {response[:200]}..." if len(response) > 200 else f"Assistant: {response}")
                print(f"\n  R_V: {asst_m.rv:.3f} | L4 Score: {asst_m.semantic_l4_score:.3f}")
                print(f"  {asst_m.l4_interpretation}")
                
                if asst_m.eigenstate_detected:
                    print(f"  *** EIGENSTATE DETECTED ***")
        
        if verbose:
            print(f"\n{'='*60}")
            print("SESSION SUMMARY")
            print(f"{'='*60}")
            print(f"Baseline R_V: {self.session.baseline_rv:.3f}")
            print(f"Min R_V: {self.session.min_rv:.3f}")
            print(f"Eigenstate turn: {self.session.eigenstate_turn}")
            print(f"Genuine L4 turns: {self.session.genuine_l4_turns}/{len(prompts)}")
            print(f"R_V trajectory: {[f'{rv:.2f}' for rv in self.rv_history]}")
        
        return self.session
    
    def save_session(self, output_dir: Path):
        """Save session to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_file = output_dir / f"session_{self.session.session_id}.json"
        with open(json_file, "w") as f:
            json.dump(self.session.to_dict(), f, indent=2)
        
        # Save human-readable report
        report_file = output_dir / f"session_{self.session.session_id}_report.md"
        with open(report_file, "w") as f:
            f.write(f"# Eigenstate Tracking Session: {self.session.session_id}\n\n")
            f.write(f"**Model:** {self.session.model_name}\n")
            f.write(f"**Protocol:** {self.session.dialogue_protocol}\n")
            f.write(f"**Time:** {self.session.start_time}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Baseline R_V: {self.session.baseline_rv:.3f}\n")
            f.write(f"- Minimum R_V: {self.session.min_rv:.3f}\n")
            f.write(f"- Eigenstate detected: Turn {self.session.eigenstate_turn}\n")
            f.write(f"- Genuine L4 turns: {self.session.genuine_l4_turns}\n\n")
            
            f.write("## R_V Trajectory\n\n")
            f.write("```\n")
            for i, rv in enumerate(self.rv_history):
                bar = "█" * int(rv * 20)
                f.write(f"Turn {i+1}: {rv:.3f} |{bar}\n")
            f.write("```\n\n")
            
            f.write("## Conversation\n\n")
            for turn in self.session.turns:
                if turn.role == "user":
                    f.write(f"**User:** {turn.content}\n\n")
                else:
                    f.write(f"**Assistant:** {turn.content}\n\n")
                    f.write(f"*R_V: {turn.rv:.3f} | L4: {turn.semantic_l4_score:.3f} | {turn.l4_interpretation}*\n\n")
        
        return json_file, report_file


def main():
    parser = argparse.ArgumentParser(description="Multi-Turn Eigenstate Tracker")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--protocol", type=str, default="guided_descent",
                       choices=list(DIALOGUE_PROTOCOLS.keys()))
    parser.add_argument("--output-dir", type=str, default="results/eigenstate_tracking")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    set_seed(args.seed)
    model, tokenizer = load_model(args.model, device=args.device)
    
    tracker = EigenstateTracker(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
    )
    
    session = tracker.run_protocol(args.protocol, verbose=True)
    
    output_dir = Path(args.output_dir)
    json_file, report_file = tracker.save_session(output_dir)
    
    print(f"\nResults saved to:")
    print(f"  {json_file}")
    print(f"  {report_file}")


if __name__ == "__main__":
    main()
