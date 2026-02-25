"""
Kaizen Engine — Toyota Production System for AI Agent Swarms

改善 (kaizen) = continuous improvement

This package provides:
- Weekly retrospective reports (kaizen events)
- Value stream mapping (kachi nagare zu)
- Andon board dashboard (real-time monitoring)
- Heijunka scheduler (level-loaded task assignment)

All data flows through AUDIT_LOG.jsonl as the single source of truth.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yaml


# Paths
REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
AUDIT_LOG_PATH = REPO_ROOT / "AUDIT_LOG.jsonl"
REPORTS_DIR = Path(__file__).parent / "reports"

# Toyota Production System term mapping
TOYOTA_TERMS: Dict[str, Dict[str, str]] = {
    "jikoku": {
        "ja": "時刻",
        "en": "Takt time measurement",
        "concept": "JIKOKU temporal audit",
    },
    "kachi_nagare_zu": {
        "ja": "価値流れ図",
        "en": "Value stream mapping",
        "concept": "Signal/Noise/Fluff taxonomy",
    },
    "andon": {
        "ja": "アンドン",
        "en": "Andon cord",
        "concept": "Circuit breaker alerts",
    },
    "jidoka": {
        "ja": "自働化",
        "en": "Automation with human touch",
        "concept": "Theater detection",
    },
    "muda": {
        "ja": "無駄",
        "en": "Pure waste",
        "concept": "Heartbeat-only loops",
    },
    "heijunka": {
        "ja": "平準化",
        "en": "Level loading",
        "concept": "Agent capability matching",
    },
    "kaizen": {
        "ja": "改善",
        "en": "Continuous improvement",
        "concept": "Weekly retrospective",
    },
    "ikko_nagashi": {
        "ja": "一個流し",
        "en": "One-piece flow",
        "concept": "Ship ONE thing",
    },
}


def load_config() -> Dict[str, Any]:
    """Load config.yaml from repo root.

    Returns:
        Configuration dictionary. Falls back to defaults if file missing.
    """
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Return default configuration when config.yaml is absent."""
    return {
        "agents": {
            "rush": {
                "name": "RUSH",
                "location": "DigitalOcean VPS",
                "type": "cloud",
                "capabilities": ["sprints", "api_work", "deployment"],
                "cost_per_hour_usd": 0.12,
                "max_concurrent_tasks": 3,
            },
            "dc": {
                "name": "DC",
                "location": "M3 Pro Local",
                "type": "local",
                "capabilities": ["local_files", "research", "analysis"],
                "cost_per_hour_usd": 0.08,
                "max_concurrent_tasks": 2,
            },
            "agent3": {
                "name": "AGENT 3",
                "location": "Oracle Cloud Free Tier",
                "type": "cloud",
                "capabilities": ["deep_reasoning", "publishing", "research"],
                "cost_per_hour_usd": 0.15,
                "max_concurrent_tasks": 4,
            },
        },
        "polling": {"interval_minutes": 15},
        "alerts": {
            "theater_loop_threshold": 5,
            "signal_ratio_minimum": 0.10,
        },
        "targets": {
            "weekly_deliverables_per_agent": 7,
            "minimum_signal_ratio": 0.30,
            "target_flow_efficiency": 0.40,
        },
    }


def load_audit_log(
    days: Optional[int] = 7,
    agent: Optional[str] = None,
    event_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load AUDIT_LOG.jsonl entries, optionally filtered.

    Args:
        days: Number of days to look back. None = all entries.
        agent: Filter by agent ID (e.g. 'rush', 'dc', 'agent3').
        event_type: Filter by event type (e.g. 'jikoku_poll', 'alert').

    Returns:
        List of audit log entries as dictionaries.
    """
    entries: List[Dict[str, Any]] = []
    cutoff = datetime.now() - timedelta(days=days) if days else None

    if not AUDIT_LOG_PATH.exists():
        return entries

    with open(AUDIT_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Time filter
                if cutoff:
                    ts_str = entry.get("timestamp", "")
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts.replace(tzinfo=None) < cutoff:
                        continue
                # Agent filter
                if agent and entry.get("agent") != agent:
                    continue
                # Event type filter
                if event_type and entry.get("event_type") != event_type:
                    continue
                entries.append(entry)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    return entries


def append_audit_log(entry: Dict[str, Any]) -> None:
    """Append a single entry to AUDIT_LOG.jsonl.

    Args:
        entry: Dictionary to write. 'timestamp' is added if missing.
    """
    if "timestamp" not in entry:
        entry["timestamp"] = datetime.now().isoformat()
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def sparkline(values: List[float], width: int = 7) -> str:
    """Generate ASCII sparkline from a list of numeric values.

    Args:
        values: Numeric values to visualize.
        width: Maximum number of characters in output.

    Returns:
        String of block characters representing the trend.
    """
    if not values:
        return "░" * width
    blocks = "▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    if mn == mx:
        return blocks[4] * min(len(values), width)
    result = ""
    for v in values[-width:]:
        idx = int((v - mn) / (mx - mn) * (len(blocks) - 1))
        result += blocks[idx]
    return result


def trend_direction(values: List[float]) -> str:
    """Determine trend direction from a series of values.

    Args:
        values: Numeric values in chronological order.

    Returns:
        One of: 'improving', 'declining', 'flat', 'insufficient_data'
    """
    if len(values) < 2:
        return "insufficient_data"
    mid = len(values) // 2
    first_half = sum(values[:mid]) / max(mid, 1)
    second_half = sum(values[mid:]) / max(len(values) - mid, 1)
    diff = second_half - first_half
    if abs(diff) < 0.05:
        return "flat"
    return "improving" if diff > 0 else "declining"


def health_score(
    signal_ratio: float,
    deliverables: int,
    alerts_critical: int,
    target_deliverables: int = 7,
) -> int:
    """Compute overall agent health score (0-100).

    Args:
        signal_ratio: Proportion of signal events (0.0-1.0).
        deliverables: Number of deliverables produced.
        alerts_critical: Number of critical alerts fired.
        target_deliverables: Weekly deliverable target.

    Returns:
        Health score from 0 (dead) to 100 (perfect).
    """
    # Signal component (40 points max)
    signal_score = min(signal_ratio / 0.5, 1.0) * 40

    # Deliverable component (40 points max)
    delivery_score = min(deliverables / max(target_deliverables, 1), 1.0) * 40

    # Alert penalty (20 points max, lose 5 per critical)
    alert_score = max(0, 20 - (alerts_critical * 5))

    return int(signal_score + delivery_score + alert_score)


def format_cost(usd: float) -> str:
    """Format USD amount for display."""
    return f"${usd:.2f}"


def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)


def get_agent_names(config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Get mapping of agent IDs to display names.

    Args:
        config: Configuration dict. Loaded from file if None.

    Returns:
        Dict mapping agent_id -> display_name.
    """
    if config is None:
        config = load_config()
    agents = config.get("agents", {})
    return {aid: ainfo.get("name", aid.upper()) for aid, ainfo in agents.items()}


# Ensure reports directory exists
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
