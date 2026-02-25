#!/usr/bin/env python3
"""
Andon Board — アンドン Dashboard

Real-time terminal dashboard showing swarm health using the Toyota
Production System Andon board metaphor. Uses the `rich` library for
beautiful terminal rendering.

Features:
- Agent status indicators (active/idle/theater/offline)
- Signal ratio bar charts per agent
- Alert history (last 10 alerts)
- Cumulative deliverables this week
- Estimated API burn rate ($/day per agent)

Usage:
    python kaizen/andon_board.py
    python kaizen/andon_board.py --refresh 30
    python kaizen/andon_board.py --no-loop
"""

import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

sys.path.insert(0, str(Path(__file__).parent.parent))
from kaizen import (
    TOYOTA_TERMS,
    format_cost,
    health_score,
    load_audit_log,
    load_config,
    sparkline,
)

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def get_agent_status(
    entries: List[Dict[str, Any]],
    agent_id: str,
    idle_threshold_hours: float = 4.0,
) -> str:
    """Determine current agent status from recent entries.

    Args:
        entries: Audit log entries (should be recent, e.g. last 24h).
        agent_id: Agent identifier.
        idle_threshold_hours: Hours without activity before marking idle.

    Returns:
        Status string: 'active', 'idle', 'theater', or 'offline'.
    """
    agent_entries = [e for e in entries if e.get("agent") == agent_id]

    if not agent_entries:
        return "offline"

    # Sort by timestamp
    agent_entries.sort(key=lambda e: e.get("timestamp", ""))
    latest = agent_entries[-1]

    # Check for recent critical alerts (theater)
    recent_alerts = [
        e for e in agent_entries
        if e.get("alert_level") == "CRITICAL"
        and e.get("event_type") == "alert"
    ]
    if recent_alerts:
        # Check if there's been signal since the last critical alert
        last_alert_ts = recent_alerts[-1].get("timestamp", "")
        signal_after = [
            e for e in agent_entries
            if e.get("timestamp", "") > last_alert_ts
            and e.get("classification") == "SIGNAL"
        ]
        if not signal_after:
            return "theater"

    # Check for idleness
    try:
        latest_ts = datetime.fromisoformat(
            latest.get("timestamp", "").replace("Z", "+00:00")
        ).replace(tzinfo=None)
        hours_since = (datetime.now() - latest_ts).total_seconds() / 3600
        if hours_since > idle_threshold_hours:
            return "idle"
    except (ValueError, TypeError):
        pass

    # Check signal in recent entries
    recent_signal = [
        e for e in agent_entries[-5:]
        if e.get("classification") == "SIGNAL"
    ]
    if recent_signal:
        return "active"

    return "idle"


STATUS_DISPLAY = {
    "active": ("🟢", "ACTIVE", "green"),
    "idle": ("🟡", "IDLE", "yellow"),
    "theater": ("🔴", "THEATER", "red"),
    "offline": ("⚪", "OFFLINE", "dim"),
}


def compute_burn_rate(
    entries: List[Dict[str, Any]], agent_id: str
) -> float:
    """Estimate daily API burn rate for an agent.

    Args:
        entries: Audit log entries (e.g. last 7 days).
        agent_id: Agent identifier.

    Returns:
        Estimated daily cost in USD.
    """
    agent_entries = [e for e in entries if e.get("agent") == agent_id]
    if not agent_entries:
        return 0.0

    total_cost = sum(e.get("estimated_cost_usd", 0) for e in agent_entries)

    # Calculate days spanned
    timestamps = []
    for e in agent_entries:
        try:
            ts = datetime.fromisoformat(
                e.get("timestamp", "").replace("Z", "+00:00")
            ).replace(tzinfo=None)
            timestamps.append(ts)
        except (ValueError, TypeError):
            continue

    if len(timestamps) < 2:
        return total_cost

    days_span = max((max(timestamps) - min(timestamps)).days, 1)
    return total_cost / days_span


def build_dashboard(config: Dict[str, Any]) -> str:
    """Build the full dashboard as a string (fallback when rich is unavailable).

    Args:
        config: Configuration dictionary.

    Returns:
        Dashboard string for terminal display.
    """
    entries_24h = load_audit_log(days=1)
    entries_7d = load_audit_log(days=7)

    agent_ids = list(config.get("agents", {}).keys())
    lines: List[str] = []

    # Header
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("")
    lines.append("╔══════════════════════════════════════════════════════════════╗")
    lines.append("║         アンドン (ANDON) BOARD — Swarm Health Dashboard     ║")
    lines.append(f"║         Last Updated: {now}                ║")
    lines.append("╚══════════════════════════════════════════════════════════════╝")
    lines.append("")

    # Agent status table
    lines.append("┌─────────┬──────────┬───────────┬────────────┬───────────┬──────────┐")
    lines.append("│ Agent   │ Status   │ Signal %  │ Delivers   │ $/day     │ Trend    │")
    lines.append("├─────────┼──────────┼───────────┼────────────┼───────────┼──────────┤")

    for aid in agent_ids:
        agent_config = config["agents"][aid]
        name = agent_config.get("name", aid.upper())
        status = get_agent_status(entries_24h, aid)
        emoji, status_text, _ = STATUS_DISPLAY[status]

        # Signal ratio (last 24h)
        agent_24h = [e for e in entries_24h if e.get("agent") == aid]
        polls = [e for e in agent_24h if e.get("event_type") == "jikoku_poll"]
        signal_pct = (
            sum(1 for p in polls if p.get("classification") == "SIGNAL") / len(polls)
            if polls
            else 0.0
        )

        # Deliverables this week
        agent_7d = [e for e in entries_7d if e.get("agent") == aid]
        deliverables = sum(e.get("deliverables_count", 0) for e in agent_7d)

        # Burn rate
        burn = compute_burn_rate(entries_7d, aid)

        # Signal trend
        daily_signals: Dict[str, List[float]] = defaultdict(list)
        for e in agent_7d:
            day = e.get("timestamp", "")[:10]
            daily_signals[day].append(e.get("signal_ratio", 0.0))
        daily_avgs = [sum(v) / len(v) for v in daily_signals.values()]
        trend = sparkline(daily_avgs)

        # Bar for signal
        bar_width = 7
        filled = int(signal_pct * bar_width)
        signal_bar = "█" * filled + "░" * (bar_width - filled)

        lines.append(
            f"│ {name:<7s} │ {emoji} {status_text:<5s} │ {signal_bar} {signal_pct:>3.0%} │ "
            f"{deliverables:>10d} │ {format_cost(burn):>9s} │ {trend:<8s} │"
        )

    lines.append("└─────────┴──────────┴───────────┴────────────┴───────────┴──────────┘")
    lines.append("")

    # Recent alerts
    lines.append("┌────────────────────────────────────────────────────────────────┐")
    lines.append("│                     Recent Alerts (Last 10)                    │")
    lines.append("├────────────────────────────────────────────────────────────────┤")

    alerts = [
        e for e in entries_7d
        if e.get("alert_level") in ("WARNING", "CRITICAL")
    ]
    alerts.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    for alert in alerts[:10]:
        ts = alert.get("timestamp", "")[:16].replace("T", " ")
        agent = alert.get("agent", "???")
        level = alert.get("alert_level", "???")
        details = alert.get("details", "")[:45]
        level_marker = "🔴" if level == "CRITICAL" else "🟡"
        lines.append(f"│ {ts} {level_marker} [{agent:>7s}] {details:<35s} │")

    if not alerts:
        lines.append("│            No alerts in the past 7 days. ✨                    │")

    lines.append("└────────────────────────────────────────────────────────────────┘")
    lines.append("")

    # Weekly summary
    total_deliverables = sum(
        e.get("deliverables_count", 0) for e in entries_7d
    )
    total_cost = sum(e.get("estimated_cost_usd", 0) for e in entries_7d)
    total_alerts_critical = sum(
        1 for e in entries_7d if e.get("alert_level") == "CRITICAL"
    )

    lines.append(f"  Weekly Summary: {total_deliverables} deliverables | "
                 f"{format_cost(total_cost)} spent | "
                 f"{total_alerts_critical} critical alerts")
    lines.append("")

    return "\n".join(lines)


def build_rich_dashboard(config: Dict[str, Any]) -> Layout:
    """Build a rich Layout dashboard.

    Args:
        config: Configuration dictionary.

    Returns:
        Rich Layout object for live rendering.
    """
    entries_24h = load_audit_log(days=1)
    entries_7d = load_audit_log(days=7)
    agent_ids = list(config.get("agents", {}).keys())

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="agents", ratio=3),
        Layout(name="alerts", ratio=2),
    )

    # Header
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = Table.grid(expand=True)
    header.add_column(justify="center")
    header.add_row(
        f"[bold]アンドン (ANDON) BOARD[/bold] — Swarm Health Dashboard — {now}"
    )
    layout["header"].update(Panel(header, style="bold white on dark_blue"))

    # Agents table
    agent_table = Table(title="Agent Status", expand=True)
    agent_table.add_column("Agent", style="bold")
    agent_table.add_column("Status", justify="center")
    agent_table.add_column("Signal %", justify="center")
    agent_table.add_column("Deliveries", justify="right")
    agent_table.add_column("$/day", justify="right")
    agent_table.add_column("Trend", justify="center")

    for aid in agent_ids:
        agent_config = config["agents"][aid]
        name = agent_config.get("name", aid.upper())
        status = get_agent_status(entries_24h, aid)
        emoji, status_text, color = STATUS_DISPLAY[status]

        # Signal
        agent_24h = [e for e in entries_24h if e.get("agent") == aid]
        polls = [e for e in agent_24h if e.get("event_type") == "jikoku_poll"]
        signal_pct = (
            sum(1 for p in polls if p.get("classification") == "SIGNAL") / len(polls)
            if polls else 0.0
        )

        # Deliverables
        agent_7d = [e for e in entries_7d if e.get("agent") == aid]
        deliverables = sum(e.get("deliverables_count", 0) for e in agent_7d)

        # Burn rate
        burn = compute_burn_rate(entries_7d, aid)

        # Trend sparkline
        daily_signals: Dict[str, List[float]] = defaultdict(list)
        for e in agent_7d:
            day = e.get("timestamp", "")[:10]
            daily_signals[day].append(e.get("signal_ratio", 0.0))
        daily_avgs = [sum(v) / len(v) for v in daily_signals.values()]
        trend = sparkline(daily_avgs)

        bar_filled = int(signal_pct * 10)
        signal_bar = f"[green]{'█' * bar_filled}[/green][dim]{'░' * (10 - bar_filled)}[/dim] {signal_pct:.0%}"

        agent_table.add_row(
            name,
            f"{emoji} [{color}]{status_text}[/{color}]",
            signal_bar,
            str(deliverables),
            format_cost(burn),
            trend,
        )

    layout["agents"].update(Panel(agent_table, border_style="blue"))

    # Alerts panel
    alerts = [
        e for e in entries_7d
        if e.get("alert_level") in ("WARNING", "CRITICAL")
    ]
    alerts.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    alert_table = Table(title="Recent Alerts", expand=True)
    alert_table.add_column("Time", style="dim")
    alert_table.add_column("Agent")
    alert_table.add_column("Details", no_wrap=False)

    for alert in alerts[:10]:
        ts = alert.get("timestamp", "")[:16].replace("T", " ")
        agent = alert.get("agent", "???")
        level = alert.get("alert_level", "")
        details = alert.get("details", "")[:50]
        style = "bold red" if level == "CRITICAL" else "yellow"
        alert_table.add_row(ts, f"[{style}]{agent}[/{style}]", details)

    if not alerts:
        alert_table.add_row("—", "—", "[green]No alerts. ✨[/green]")

    layout["alerts"].update(Panel(alert_table, border_style="red"))

    # Footer
    total_deliverables = sum(e.get("deliverables_count", 0) for e in entries_7d)
    total_cost = sum(e.get("estimated_cost_usd", 0) for e in entries_7d)
    total_critical = sum(1 for e in entries_7d if e.get("alert_level") == "CRITICAL")

    footer = Table.grid(expand=True)
    footer.add_column(justify="center")
    footer.add_row(
        f"[bold]Weekly:[/bold] {total_deliverables} deliverables | "
        f"{format_cost(total_cost)} spent | "
        f"{total_critical} critical alerts | "
        f"[dim]改善 — continuous improvement[/dim]"
    )
    layout["footer"].update(Panel(footer, style="dim"))

    return layout


@click.command()
@click.option("--refresh", default=60, help="Refresh interval in seconds")
@click.option("--no-loop", is_flag=True, help="Print once and exit (no auto-refresh)")
@click.option("--plain", is_flag=True, help="Use plain ASCII output (no rich)")
def main(refresh: int, no_loop: bool, plain: bool) -> None:
    """Launch the Andon Board dashboard."""
    config = load_config()

    use_rich = HAS_RICH and not plain

    if no_loop:
        if use_rich:
            console = Console()
            layout = build_rich_dashboard(config)
            console.print(layout)
        else:
            click.echo(build_dashboard(config))
        return

    if use_rich:
        console = Console()
        click.echo(f"🏭 Andon Board starting (refresh every {refresh}s, Ctrl+C to stop)")
        try:
            with Live(
                build_rich_dashboard(config),
                console=console,
                refresh_per_second=0.5,
                screen=True,
            ) as live:
                while True:
                    time.sleep(refresh)
                    live.update(build_rich_dashboard(config))
        except KeyboardInterrupt:
            click.echo("\n👋 Andon Board stopped.")
    else:
        click.echo(f"🏭 Andon Board starting (refresh every {refresh}s, Ctrl+C to stop)")
        try:
            while True:
                # Clear screen
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.flush()
                click.echo(build_dashboard(config))
                time.sleep(refresh)
        except KeyboardInterrupt:
            click.echo("\n👋 Andon Board stopped.")


if __name__ == "__main__":
    main()
