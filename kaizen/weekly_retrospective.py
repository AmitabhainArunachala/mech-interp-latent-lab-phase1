#!/usr/bin/env python3
"""
Weekly Retrospective — 改善 (Kaizen) Event Generator

Reads AUDIT_LOG.jsonl for the past 7 days and generates a comprehensive
kaizen report with per-agent scorecards, waste identification, and
trend analysis using Toyota Production System terminology.

Usage:
    python kaizen/weekly_retrospective.py
    python kaizen/weekly_retrospective.py --days 14
    python kaizen/weekly_retrospective.py --output custom_report.md
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from kaizen import (
    REPORTS_DIR,
    TOYOTA_TERMS,
    format_cost,
    health_score,
    load_audit_log,
    load_config,
    sparkline,
    trend_direction,
)


def compute_agent_metrics(
    entries: List[Dict[str, Any]], agent_id: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute comprehensive metrics for a single agent.

    Args:
        entries: All audit log entries for the period.
        agent_id: Agent identifier (e.g. 'rush', 'dc', 'agent3').
        config: Full configuration dictionary.

    Returns:
        Dictionary with computed metrics for the agent.
    """
    agent_entries = [e for e in entries if e.get("agent") == agent_id]

    if not agent_entries:
        return {
            "agent_id": agent_id,
            "name": config.get("agents", {}).get(agent_id, {}).get("name", agent_id.upper()),
            "total_polls": 0,
            "signal_count": 0,
            "noise_count": 0,
            "fluff_count": 0,
            "signal_ratio": 0.0,
            "noise_ratio": 0.0,
            "fluff_ratio": 0.0,
            "deliverables_count": 0,
            "deliverables": [],
            "critical_alerts": 0,
            "warning_alerts": 0,
            "sessions": 0,
            "total_cost_usd": 0.0,
            "cost_per_deliverable": 0.0,
            "signal_ratios_daily": [],
            "signal_trend": "insufficient_data",
            "signal_sparkline": "░░░░░░░",
            "health": 0,
            "theater_days": 0,
            "value_added_ratio": 0.0,
        }

    # Classification counts
    signal_count = sum(1 for e in agent_entries if e.get("classification") == "SIGNAL")
    noise_count = sum(1 for e in agent_entries if e.get("classification") == "NOISE")
    fluff_count = sum(1 for e in agent_entries if e.get("classification") == "FLUFF")
    total = signal_count + noise_count + fluff_count

    # Deliverables
    all_deliverables: List[str] = []
    for e in agent_entries:
        all_deliverables.extend(e.get("deliverables", []))
    deliverables_count = len(all_deliverables)

    # Alerts
    critical_alerts = sum(
        1 for e in agent_entries if e.get("alert_level") == "CRITICAL"
    )
    warning_alerts = sum(
        1 for e in agent_entries if e.get("alert_level") == "WARNING"
    )

    # Sessions
    sessions = sum(
        1 for e in agent_entries if e.get("event_type") == "session_start"
    )

    # Cost
    total_cost = sum(e.get("estimated_cost_usd", 0) for e in agent_entries)
    cost_per_deliverable = (
        total_cost / deliverables_count if deliverables_count > 0 else float("inf")
    )

    # Daily signal ratios for trend
    daily_signals: Dict[str, List[float]] = defaultdict(list)
    for e in agent_entries:
        day = e.get("timestamp", "")[:10]
        sr = e.get("signal_ratio", 0.0)
        daily_signals[day].append(sr)

    daily_avg_signals = [
        sum(v) / len(v) for v in daily_signals.values()
    ]

    # Theater detection: consecutive days with 0 deliverables
    theater_days = 0
    days_sorted = sorted(daily_signals.keys())
    for day in reversed(days_sorted):
        day_entries = [
            e for e in agent_entries if e.get("timestamp", "").startswith(day)
        ]
        day_deliverables = sum(e.get("deliverables_count", 0) for e in day_entries)
        day_signal = sum(e.get("signal_ratio", 0) for e in day_entries)
        if day_deliverables == 0 and day_signal < 0.1:
            theater_days += 1
        else:
            break

    # Ratios
    signal_ratio = signal_count / total if total > 0 else 0.0
    noise_ratio = noise_count / total if total > 0 else 0.0
    fluff_ratio = fluff_count / total if total > 0 else 0.0

    # Value-added ratio: signal polls / total polls (excluding session start/end)
    poll_entries = [
        e for e in agent_entries if e.get("event_type") == "jikoku_poll"
    ]
    value_added = (
        sum(1 for e in poll_entries if e.get("classification") == "SIGNAL")
        / len(poll_entries)
        if poll_entries
        else 0.0
    )

    target_deliverables = config.get("targets", {}).get(
        "weekly_deliverables_per_agent", 7
    )

    return {
        "agent_id": agent_id,
        "name": config.get("agents", {}).get(agent_id, {}).get("name", agent_id.upper()),
        "total_polls": total,
        "signal_count": signal_count,
        "noise_count": noise_count,
        "fluff_count": fluff_count,
        "signal_ratio": signal_ratio,
        "noise_ratio": noise_ratio,
        "fluff_ratio": fluff_ratio,
        "deliverables_count": deliverables_count,
        "deliverables": all_deliverables,
        "critical_alerts": critical_alerts,
        "warning_alerts": warning_alerts,
        "sessions": sessions,
        "total_cost_usd": total_cost,
        "cost_per_deliverable": cost_per_deliverable,
        "signal_ratios_daily": daily_avg_signals,
        "signal_trend": trend_direction(daily_avg_signals),
        "signal_sparkline": sparkline(daily_avg_signals),
        "health": health_score(
            signal_ratio, deliverables_count, critical_alerts, target_deliverables
        ),
        "theater_days": theater_days,
        "value_added_ratio": value_added,
    }


def identify_wastes(
    agent_metrics: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Identify top wastes across all agents using Toyota terminology.

    Args:
        agent_metrics: List of per-agent metric dictionaries.

    Returns:
        List of waste items with toyota_term, description, agent, and severity.
    """
    wastes: List[Dict[str, str]] = []

    for m in agent_metrics:
        # Theater loop = Muda (pure waste)
        if m["theater_days"] >= 3:
            wastes.append(
                {
                    "toyota_term": f"無駄 (Muda) — {TOYOTA_TERMS['muda']['en']}",
                    "description": (
                        f"{m['name']} has been in theater loop for "
                        f"{m['theater_days']} consecutive days. "
                        f"0 deliverables, {m['critical_alerts']} critical alerts."
                    ),
                    "agent": m["name"],
                    "severity": "CRITICAL",
                }
            )

        # High fluff ratio = Jidoka failure
        if m["fluff_ratio"] > 0.70:
            wastes.append(
                {
                    "toyota_term": f"自働化 (Jidoka) — {TOYOTA_TERMS['jidoka']['en']}",
                    "description": (
                        f"{m['name']} fluff ratio at {m['fluff_ratio']:.0%}. "
                        f"Automation not detecting waste early enough."
                    ),
                    "agent": m["name"],
                    "severity": "HIGH",
                }
            )

        # Low deliverables with high cost = poor flow
        if m["deliverables_count"] < 3 and m["total_cost_usd"] > 1.0:
            wastes.append(
                {
                    "toyota_term": f"一個流し (Ikko Nagashi) — {TOYOTA_TERMS['ikko_nagashi']['en']}",
                    "description": (
                        f"{m['name']} spent {format_cost(m['total_cost_usd'])} "
                        f"but only produced {m['deliverables_count']} deliverables. "
                        f"Cost/deliverable: {format_cost(m['cost_per_deliverable'])}."
                    ),
                    "agent": m["name"],
                    "severity": "MEDIUM",
                }
            )

        # Sprint with no follow-up
        if m["signal_trend"] == "declining":
            wastes.append(
                {
                    "toyota_term": f"平準化 (Heijunka) — {TOYOTA_TERMS['heijunka']['en']}",
                    "description": (
                        f"{m['name']} signal trend is declining "
                        f"({m['signal_sparkline']}). "
                        f"Work not level-loaded across the week."
                    ),
                    "agent": m["name"],
                    "severity": "MEDIUM",
                }
            )

    # Sort by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    wastes.sort(key=lambda w: severity_order.get(w["severity"], 99))
    return wastes


def detect_patterns(
    entries: List[Dict[str, Any]],
    agent_metrics: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[str]:
    """Detect notable patterns across the swarm.

    Args:
        entries: All audit log entries.
        agent_metrics: Per-agent metrics.
        config: Configuration.

    Returns:
        List of pattern description strings.
    """
    patterns: List[str] = []

    for m in agent_metrics:
        if m["theater_days"] > 0:
            patterns.append(
                f"- {m['name']} has been in theater loop for "
                f"**{m['theater_days']} consecutive days** "
                f"(0 signal, {m['fluff_ratio']:.0%} fluff)"
            )

        if m["signal_trend"] == "improving":
            patterns.append(
                f"- {m['name']} signal trend is **improving** "
                f"{m['signal_sparkline']}"
            )

        if m["deliverables_count"] >= 7:
            target = config.get("targets", {}).get("weekly_deliverables_per_agent", 7)
            patterns.append(
                f"- {m['name']} exceeded weekly target: "
                f"**{m['deliverables_count']}/{target}** deliverables"
            )

        if m["deliverables_count"] > 0 and m["cost_per_deliverable"] < 0.50:
            patterns.append(
                f"- {m['name']} cost-efficient: "
                f"**{format_cost(m['cost_per_deliverable'])}** per deliverable"
            )

    return patterns


def generate_report(
    entries: List[Dict[str, Any]],
    config: Dict[str, Any],
    days: int = 7,
    report_date: Optional[str] = None,
) -> str:
    """Generate the full kaizen report as Markdown.

    Args:
        entries: Filtered audit log entries.
        config: Configuration dictionary.
        days: Number of days covered.
        report_date: Override date string for the report.

    Returns:
        Complete Markdown report string.
    """
    today = report_date or datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Week number
    week_num = datetime.now().isocalendar()[1]

    # Compute per-agent metrics
    agent_ids = list(config.get("agents", {}).keys())
    agent_metrics = [
        compute_agent_metrics(entries, aid, config) for aid in agent_ids
    ]

    # Overall health
    healths = [m["health"] for m in agent_metrics if m["total_polls"] > 0]
    overall_health = int(sum(healths) / len(healths)) if healths else 0

    # Total deliverables and cost
    total_deliverables = sum(m["deliverables_count"] for m in agent_metrics)
    total_cost = sum(m["total_cost_usd"] for m in agent_metrics)
    total_alerts = sum(m["critical_alerts"] for m in agent_metrics)

    # Wastes and patterns
    wastes = identify_wastes(agent_metrics)
    patterns = detect_patterns(entries, agent_metrics, config)

    # Build report
    lines: List[str] = []

    # Header
    lines.append(f"# 改善 (Kaizen) Weekly Report — Week {week_num}")
    lines.append("")
    lines.append(f"**Date Range:** {start_date} → {today}")
    lines.append(f"**Overall Swarm Health:** {overall_health}/100")
    lines.append(f"**Total Deliverables:** {total_deliverables}")
    lines.append(f"**Total Cost:** {format_cost(total_cost)}")
    lines.append(f"**Critical Alerts:** {total_alerts}")
    lines.append("")

    # Health bar
    bar_filled = "█" * (overall_health // 5)
    bar_empty = "░" * (20 - overall_health // 5)
    lines.append(f"Health: [{bar_filled}{bar_empty}] {overall_health}%")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-agent scorecards
    lines.append("## Agent Scorecards")
    lines.append("")

    for m in agent_metrics:
        status_emoji = (
            "🔴" if m["health"] < 25
            else "🟡" if m["health"] < 50
            else "🟢"
        )
        lines.append(f"### {status_emoji} {m['name']} ({m['agent_id']})")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Health Score | **{m['health']}/100** |")
        lines.append(f"| Signal Ratio | {m['signal_ratio']:.0%} ({m['signal_count']}/{m['total_polls']}) |")
        lines.append(f"| Noise Ratio | {m['noise_ratio']:.0%} ({m['noise_count']}/{m['total_polls']}) |")
        lines.append(f"| Fluff Ratio | {m['fluff_ratio']:.0%} ({m['fluff_count']}/{m['total_polls']}) |")
        lines.append(f"| Deliverables | **{m['deliverables_count']}** |")
        lines.append(f"| Sessions | {m['sessions']} |")
        lines.append(f"| Critical Alerts | {m['critical_alerts']} |")
        lines.append(f"| Total Cost | {format_cost(m['total_cost_usd'])} |")
        lines.append(f"| Cost/Deliverable | {format_cost(m['cost_per_deliverable']) if m['deliverables_count'] > 0 else 'N/A (0 deliverables)'} |")
        lines.append(f"| Signal Trend | {m['signal_trend']} {m['signal_sparkline']} |")
        lines.append(f"| Value-Added Ratio | {m['value_added_ratio']:.0%} |")

        if m["theater_days"] > 0:
            lines.append(f"| **Theater Days** | **{m['theater_days']}** ⚠️ |")

        if m["deliverables"]:
            lines.append("")
            lines.append(f"**Deliverables:** {', '.join(f'`{d}`' for d in m['deliverables'])}")

        lines.append("")

    lines.append("---")
    lines.append("")

    # Top Wastes
    lines.append("## Top Wastes Identified")
    lines.append("")

    for i, w in enumerate(wastes[:5], 1):
        lines.append(f"### {i}. [{w['severity']}] {w['toyota_term']}")
        lines.append(f"**Agent:** {w['agent']}")
        lines.append(f"**Details:** {w['description']}")
        lines.append("")

    if not wastes:
        lines.append("*No significant wastes detected. 素晴らしい！*")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Patterns
    lines.append("## Detected Patterns")
    lines.append("")
    if patterns:
        for p in patterns:
            lines.append(p)
    else:
        lines.append("*No notable patterns detected.*")
    lines.append("")

    lines.append("---")
    lines.append("")

    # Recommended Actions
    lines.append("## Recommended Actions for Next Week")
    lines.append("")

    action_num = 1
    for m in agent_metrics:
        if m["theater_days"] >= 3:
            lines.append(
                f"{action_num}. **STOP** {m['name']} sessions until root cause "
                f"of theater loop is identified. Reassign tasks to productive agents."
            )
            action_num += 1

        if m["signal_trend"] == "declining":
            lines.append(
                f"{action_num}. **INVESTIGATE** {m['name']} declining signal. "
                f"Check task assignment quality and agent capability match."
            )
            action_num += 1

        if m["deliverables_count"] >= 5 and m["signal_trend"] != "declining":
            lines.append(
                f"{action_num}. **MAINTAIN** {m['name']} current workload. "
                f"Consider increasing task complexity."
            )
            action_num += 1

    if total_cost > 0 and total_deliverables > 0:
        efficiency = total_deliverables / total_cost
        lines.append(
            f"{action_num}. **MONITOR** swarm cost efficiency: "
            f"{efficiency:.1f} deliverables/$ (target: >2.0/$ )"
        )
        action_num += 1

    lines.append("")
    lines.append("---")
    lines.append("")

    # Toyota mapping reference
    lines.append("## Toyota Production System Reference")
    lines.append("")
    lines.append("| AI Agent Concept | Toyota Equivalent | Japanese Term |")
    lines.append("|-----------------|-------------------|---------------|")
    for key, term in TOYOTA_TERMS.items():
        lines.append(f"| {term['concept']} | {term['en']} | {term['ja']} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"*Generated by META_META_KNOWER Kaizen Engine on "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    )
    lines.append("")
    lines.append("*一個流し — Ship ONE thing, then the next.*")

    return "\n".join(lines)


@click.command()
@click.option("--days", default=7, help="Number of days to analyze")
@click.option("--output", default=None, help="Custom output path for the report")
@click.option("--dry-run", is_flag=True, help="Print report to stdout without saving")
def main(days: int, output: Optional[str], dry_run: bool) -> None:
    """Generate weekly kaizen retrospective report."""
    config = load_config()
    entries = load_audit_log(days=days)

    if not entries:
        click.echo("⚠️  No audit log entries found. Is AUDIT_LOG.jsonl present?")
        click.echo(f"   Expected at: {Path(__file__).parent.parent / 'AUDIT_LOG.jsonl'}")
        return

    click.echo(f"📊 Analyzing {len(entries)} audit log entries over {days} days...")

    report = generate_report(entries, config, days=days)

    if dry_run:
        click.echo(report)
        return

    # Determine output path
    if output:
        report_path = Path(output)
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        report_path = REPORTS_DIR / f"KAIZEN_REPORT_{today}.md"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    click.echo(f"✅ Kaizen report saved to: {report_path}")

    # Also save machine-readable summary
    summary_path = report_path.with_suffix(".json")
    agent_ids = list(config.get("agents", {}).keys())
    summary = {
        "report_date": datetime.now().isoformat(),
        "days_analyzed": days,
        "entries_analyzed": len(entries),
        "agents": {
            aid: compute_agent_metrics(entries, aid, config) for aid in agent_ids
        },
        "wastes": identify_wastes(
            [compute_agent_metrics(entries, aid, config) for aid in agent_ids]
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    click.echo(f"✅ Machine-readable summary: {summary_path}")


if __name__ == "__main__":
    main()
