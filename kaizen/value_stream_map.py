#!/usr/bin/env python3
"""
Value Stream Map — 価値流れ図 (Kachi Nagare Zu)

Takes an agent's JIKOKU log for a time period and maps every action
into VALUE-ADDED / NECESSARY-NON-VALUE / PURE-WASTE categories.
Outputs a visual ASCII value stream map showing where time goes.

Usage:
    python kaizen/value_stream_map.py --agent rush --date 2026-02-08
    python kaizen/value_stream_map.py --agent dc --date today
    python kaizen/value_stream_map.py --agent agent3 --days 3
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

sys.path.insert(0, str(Path(__file__).parent.parent))
from kaizen import (
    TOYOTA_TERMS,
    format_cost,
    load_audit_log,
    load_config,
    parse_timestamp,
    sparkline,
)


# Classification constants
VALUE_ADDED = "VALUE-ADDED"
NECESSARY_NON_VALUE = "NECESSARY-NON-VALUE"
PURE_WASTE = "PURE-WASTE"

# Color codes for terminal
COLORS = {
    VALUE_ADDED: "\033[92m",         # Green
    NECESSARY_NON_VALUE: "\033[93m", # Yellow
    PURE_WASTE: "\033[91m",          # Red
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
}


def classify_action(entry: Dict[str, Any]) -> str:
    """Classify an audit log entry into value stream categories.

    Args:
        entry: Single audit log entry dictionary.

    Returns:
        One of VALUE-ADDED, NECESSARY-NON-VALUE, or PURE-WASTE.
    """
    classification = entry.get("classification", "FLUFF")
    event_type = entry.get("event_type", "")
    signal_ratio = entry.get("signal_ratio", 0.0)
    deliverables = entry.get("deliverables_count", 0)

    # Direct signal with deliverables = value-added
    if classification == "SIGNAL" and deliverables > 0:
        return VALUE_ADDED

    # Signal without deliverables (good work in progress)
    if classification == "SIGNAL":
        return VALUE_ADDED

    # Session start/end are necessary overhead
    if event_type in ("session_start", "session_end"):
        return NECESSARY_NON_VALUE

    # Noise with some signal = necessary non-value (debugging, planning)
    if classification == "NOISE" and signal_ratio > 0.1:
        return NECESSARY_NON_VALUE

    # Pure noise (config, setup) = necessary non-value
    if classification == "NOISE":
        return NECESSARY_NON_VALUE

    # Fluff = pure waste (heartbeat theater)
    if classification == "FLUFF":
        return PURE_WASTE

    # Alerts are necessary overhead
    if event_type == "alert":
        return NECESSARY_NON_VALUE

    return PURE_WASTE


def compute_flow_metrics(
    entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute value stream flow metrics.

    Args:
        entries: List of audit log entries for one agent/period.

    Returns:
        Dictionary with flow efficiency metrics.
    """
    if not entries:
        return {
            "value_added_count": 0,
            "necessary_non_value_count": 0,
            "pure_waste_count": 0,
            "total_count": 0,
            "flow_efficiency": 0.0,
            "value_added_pct": 0.0,
            "necessary_pct": 0.0,
            "waste_pct": 0.0,
            "total_cost": 0.0,
            "wasted_cost": 0.0,
            "classifications": [],
        }

    classifications = [(e, classify_action(e)) for e in entries]

    va_count = sum(1 for _, c in classifications if c == VALUE_ADDED)
    nnv_count = sum(1 for _, c in classifications if c == NECESSARY_NON_VALUE)
    pw_count = sum(1 for _, c in classifications if c == PURE_WASTE)
    total = len(classifications)

    total_cost = sum(e.get("estimated_cost_usd", 0) for e in entries)
    wasted_cost = sum(
        e.get("estimated_cost_usd", 0)
        for e, c in classifications
        if c == PURE_WASTE
    )

    # Flow efficiency = value-added time / total elapsed time
    flow_efficiency = va_count / total if total > 0 else 0.0

    return {
        "value_added_count": va_count,
        "necessary_non_value_count": nnv_count,
        "pure_waste_count": pw_count,
        "total_count": total,
        "flow_efficiency": flow_efficiency,
        "value_added_pct": va_count / total if total > 0 else 0.0,
        "necessary_pct": nnv_count / total if total > 0 else 0.0,
        "waste_pct": pw_count / total if total > 0 else 0.0,
        "total_cost": total_cost,
        "wasted_cost": wasted_cost,
        "classifications": classifications,
    }


def render_stream_bar(
    va_pct: float, nnv_pct: float, pw_pct: float, width: int = 60
) -> str:
    """Render a colored horizontal bar showing value stream proportions.

    Args:
        va_pct: Value-added percentage (0-1).
        nnv_pct: Necessary non-value percentage (0-1).
        pw_pct: Pure waste percentage (0-1).
        width: Total bar width in characters.

    Returns:
        Colored bar string.
    """
    va_chars = int(va_pct * width)
    nnv_chars = int(nnv_pct * width)
    pw_chars = width - va_chars - nnv_chars  # Remainder goes to waste

    bar = (
        f"{COLORS[VALUE_ADDED]}{'█' * va_chars}"
        f"{COLORS[NECESSARY_NON_VALUE]}{'▓' * nnv_chars}"
        f"{COLORS[PURE_WASTE]}{'░' * pw_chars}"
        f"{COLORS['RESET']}"
    )
    return bar


def render_timeline(
    classifications: List[Tuple[Dict[str, Any], str]],
    width: int = 72,
) -> List[str]:
    """Render a timeline showing action-by-action classification.

    Args:
        classifications: List of (entry, classification) tuples.
        width: Maximum line width.

    Returns:
        List of formatted timeline strings.
    """
    lines: List[str] = []
    symbol_map = {
        VALUE_ADDED: f"{COLORS[VALUE_ADDED]}●{COLORS['RESET']}",
        NECESSARY_NON_VALUE: f"{COLORS[NECESSARY_NON_VALUE]}◐{COLORS['RESET']}",
        PURE_WASTE: f"{COLORS[PURE_WASTE]}○{COLORS['RESET']}",
    }

    for entry, cls in classifications:
        ts = entry.get("timestamp", "")
        time_str = ts[11:16] if len(ts) > 16 else "??:??"
        symbol = symbol_map.get(cls, "?")
        details = entry.get("details", "")[:50]
        deliverables = entry.get("deliverables", [])
        cost = entry.get("estimated_cost_usd", 0)

        deliverable_str = ""
        if deliverables:
            deliverable_str = f" → {', '.join(deliverables)}"

        line = (
            f"  {time_str} {symbol} [{cls:^22s}] "
            f"{details}{deliverable_str} "
            f"({format_cost(cost)})"
        )
        lines.append(line)

    return lines


def render_ascii_vsm(
    agent_name: str,
    metrics: Dict[str, Any],
    date_range: str,
) -> str:
    """Render the complete ASCII value stream map.

    Args:
        agent_name: Display name of the agent.
        metrics: Flow metrics dictionary.
        date_range: String describing the analyzed period.

    Returns:
        Complete ASCII art value stream map as a string.
    """
    lines: List[str] = []

    # Header
    lines.append("")
    lines.append(f"{COLORS['BOLD']}╔{'═' * 70}╗{COLORS['RESET']}")
    lines.append(f"{COLORS['BOLD']}║  価値流れ図 (Value Stream Map) — {agent_name:<36s}║{COLORS['RESET']}")
    lines.append(f"{COLORS['BOLD']}║  Period: {date_range:<58s}║{COLORS['RESET']}")
    lines.append(f"{COLORS['BOLD']}╚{'═' * 70}╝{COLORS['RESET']}")
    lines.append("")

    # Summary metrics
    lines.append(f"  {COLORS['BOLD']}Flow Efficiency:{COLORS['RESET']} {metrics['flow_efficiency']:.1%}")
    lines.append(
        f"  Total Actions: {metrics['total_count']}  |  "
        f"Total Cost: {format_cost(metrics['total_cost'])}  |  "
        f"Wasted: {format_cost(metrics['wasted_cost'])}"
    )
    lines.append("")

    # Value stream bar
    lines.append(f"  {COLORS['BOLD']}Value Stream:{COLORS['RESET']}")
    bar = render_stream_bar(
        metrics["value_added_pct"],
        metrics["necessary_pct"],
        metrics["waste_pct"],
    )
    lines.append(f"  {bar}")
    lines.append(
        f"  {COLORS[VALUE_ADDED]}█ Value-Added: {metrics['value_added_pct']:.0%}{COLORS['RESET']}  "
        f"{COLORS[NECESSARY_NON_VALUE]}▓ Necessary: {metrics['necessary_pct']:.0%}{COLORS['RESET']}  "
        f"{COLORS[PURE_WASTE]}░ Waste: {metrics['waste_pct']:.0%}{COLORS['RESET']}"
    )
    lines.append("")

    # Timeline
    if metrics["classifications"]:
        lines.append(f"  {COLORS['BOLD']}Action Timeline:{COLORS['RESET']}")
        lines.append(f"  {'─' * 68}")
        timeline = render_timeline(metrics["classifications"])
        lines.extend(timeline)
        lines.append(f"  {'─' * 68}")
    lines.append("")

    # Diagnosis
    lines.append(f"  {COLORS['BOLD']}Diagnosis:{COLORS['RESET']}")
    if metrics["flow_efficiency"] >= 0.5:
        lines.append(
            f"  ✅ Flow efficiency is healthy ({metrics['flow_efficiency']:.0%}). "
            f"Agent is producing value."
        )
    elif metrics["flow_efficiency"] >= 0.2:
        lines.append(
            f"  ⚠️  Flow efficiency is below target ({metrics['flow_efficiency']:.0%}). "
            f"Review necessary-non-value activities for reduction opportunities."
        )
    else:
        lines.append(
            f"  🔴 Flow efficiency is critically low ({metrics['flow_efficiency']:.0%}). "
            f"Agent is primarily producing waste. Intervention required."
        )

    if metrics["wasted_cost"] > 0:
        waste_pct_of_cost = (
            metrics["wasted_cost"] / metrics["total_cost"]
            if metrics["total_cost"] > 0
            else 0
        )
        lines.append(
            f"  💸 {format_cost(metrics['wasted_cost'])} wasted "
            f"({waste_pct_of_cost:.0%} of total spend)"
        )

    lines.append("")
    lines.append(
        f"  {COLORS['DIM']}Legend: "
        f"{COLORS[VALUE_ADDED]}● Value-Added{COLORS['RESET']}  "
        f"{COLORS[NECESSARY_NON_VALUE]}◐ Necessary{COLORS['RESET']}  "
        f"{COLORS[PURE_WASTE]}○ Waste{COLORS['RESET']}"
    )
    lines.append("")

    return "\n".join(lines)


@click.command()
@click.option("--agent", required=True, help="Agent ID (rush, dc, agent3)")
@click.option(
    "--date",
    default="today",
    help="Date to analyze (YYYY-MM-DD or 'today')",
)
@click.option("--days", default=1, help="Number of days to analyze")
@click.option("--no-color", is_flag=True, help="Disable colored output")
@click.option("--markdown", is_flag=True, help="Output as markdown instead of terminal art")
def main(
    agent: str,
    date: str,
    days: int,
    no_color: bool,
    markdown: bool,
) -> None:
    """Generate value stream map for an agent."""
    if no_color:
        for key in COLORS:
            COLORS[key] = ""

    config = load_config()

    # Validate agent
    if agent not in config.get("agents", {}):
        click.echo(f"❌ Unknown agent: {agent}")
        click.echo(f"   Available: {', '.join(config.get('agents', {}).keys())}")
        return

    agent_name = config["agents"][agent].get("name", agent.upper())

    # Parse date
    if date == "today":
        target_date = datetime.now()
    else:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            click.echo(f"❌ Invalid date format: {date}. Use YYYY-MM-DD or 'today'.")
            return

    # Load entries
    entries = load_audit_log(days=days + 1, agent=agent)

    # Filter to date range
    start = target_date - timedelta(days=days - 1)
    start = start.replace(hour=0, minute=0, second=0)
    end = target_date.replace(hour=23, minute=59, second=59)

    filtered = []
    for e in entries:
        try:
            ts = parse_timestamp(e["timestamp"])
            if start <= ts <= end:
                filtered.append(e)
        except (KeyError, ValueError):
            continue

    if not filtered:
        click.echo(f"⚠️  No entries found for {agent_name} on {date}")
        return

    date_range = (
        f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        if days > 1
        else target_date.strftime("%Y-%m-%d")
    )

    # Compute metrics
    metrics = compute_flow_metrics(filtered)

    if markdown:
        # Markdown output for reports
        md_lines = [
            f"# Value Stream Map — {agent_name}",
            f"**Period:** {date_range}",
            "",
            f"## Flow Efficiency: {metrics['flow_efficiency']:.1%}",
            "",
            "| Category | Count | Percentage | Cost |",
            "|----------|-------|------------|------|",
            f"| Value-Added | {metrics['value_added_count']} | {metrics['value_added_pct']:.0%} | — |",
            f"| Necessary Non-Value | {metrics['necessary_non_value_count']} | {metrics['necessary_pct']:.0%} | — |",
            f"| Pure Waste | {metrics['pure_waste_count']} | {metrics['waste_pct']:.0%} | {format_cost(metrics['wasted_cost'])} |",
            f"| **Total** | **{metrics['total_count']}** | **100%** | **{format_cost(metrics['total_cost'])}** |",
            "",
        ]
        click.echo("\n".join(md_lines))
    else:
        # Terminal art output
        vsm = render_ascii_vsm(agent_name, metrics, date_range)
        click.echo(vsm)


if __name__ == "__main__":
    main()
