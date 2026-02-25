#!/usr/bin/env python3
"""
Heijunka Scheduler — 平準化 (Level Loading) for Agent Swarms

Given pending tasks and agent capability profiles, assigns tasks to agents
based on current load, capability match, priority, and deadline.
Outputs HEIJUNKA_BOARD.md and per-agent INTERVENTION.md files.

Usage:
    python kaizen/heijunka_scheduler.py --tasks pending_tasks.yaml
    python kaizen/heijunka_scheduler.py --tasks pending_tasks.yaml --dry-run
    python kaizen/heijunka_scheduler.py  # Uses default sample tasks
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from kaizen import (
    REPO_ROOT,
    TOYOTA_TERMS,
    format_cost,
    load_audit_log,
    load_config,
)


# Default sample tasks when no YAML is provided
DEFAULT_TASKS: List[Dict[str, Any]] = [
    {
        "id": "T001",
        "title": "Deploy Agent 3 to Oracle Cloud",
        "priority": 1,
        "deadline": "2026-02-11",
        "required_capabilities": ["deployment", "deep_reasoning"],
        "estimated_hours": 4,
        "description": "Set up Agent 3 on Oracle Cloud Free Tier with OpenClaw + Claude Opus 4.6",
    },
    {
        "id": "T002",
        "title": "Write weekly blog post on R_V metric",
        "priority": 2,
        "deadline": "2026-02-14",
        "required_capabilities": ["publishing", "research"],
        "estimated_hours": 3,
        "description": "Technical blog post explaining R_V metric methodology and results",
    },
    {
        "id": "T003",
        "title": "Fix DC theater loop root cause",
        "priority": 1,
        "deadline": "2026-02-10",
        "required_capabilities": ["analysis", "local_files"],
        "estimated_hours": 2,
        "description": "Investigate why DC has 26 consecutive zero-value sessions",
    },
    {
        "id": "T004",
        "title": "Build webhook notification system",
        "priority": 3,
        "deadline": "2026-02-16",
        "required_capabilities": ["api_work", "deployment"],
        "estimated_hours": 3,
        "description": "Webhook handler for alert notifications to Slack/Discord",
    },
    {
        "id": "T005",
        "title": "Cross-architecture R_V validation on Llama",
        "priority": 2,
        "deadline": "2026-02-15",
        "required_capabilities": ["deep_reasoning", "research"],
        "estimated_hours": 6,
        "description": "Run R_V validation experiment on Llama-3.2-8B",
    },
    {
        "id": "T006",
        "title": "Create kaizen report automation cron",
        "priority": 2,
        "deadline": "2026-02-12",
        "required_capabilities": ["sprints", "batch_processing"],
        "estimated_hours": 2,
        "description": "Set up weekly cron job for automatic kaizen report generation",
    },
    {
        "id": "T007",
        "title": "Package Kaizen Swarm Optimizer for ClawHub",
        "priority": 3,
        "deadline": "2026-02-20",
        "required_capabilities": ["code_generation", "publishing"],
        "estimated_hours": 8,
        "description": "Package the monitoring + kaizen tools as a ClawHub skill",
    },
    {
        "id": "T008",
        "title": "Audit RUSH sprint effectiveness",
        "priority": 3,
        "deadline": "2026-02-14",
        "required_capabilities": ["analysis", "research"],
        "estimated_hours": 2,
        "description": "Analyze RUSH's sprint patterns and deliverable quality over past month",
    },
]


def compute_agent_load(
    entries: List[Dict[str, Any]], agent_id: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute current workload for an agent.

    Args:
        entries: Recent audit log entries.
        agent_id: Agent identifier.
        config: Configuration dictionary.

    Returns:
        Dictionary with load metrics.
    """
    agent_entries = [e for e in entries if e.get("agent") == agent_id]
    agent_config = config.get("agents", {}).get(agent_id, {})

    # Count active sessions in last 24h
    recent = [
        e for e in agent_entries
        if e.get("event_type") == "session_start"
    ]

    # Theater status
    is_theater = any(
        e.get("alert_level") == "CRITICAL" and "THEATER" in e.get("details", "")
        for e in agent_entries
    )

    # Compute hours of activity
    total_polls = len([
        e for e in agent_entries if e.get("event_type") == "jikoku_poll"
    ])
    estimated_active_hours = total_polls * 0.25  # 15 min per poll

    max_tasks = agent_config.get("max_concurrent_tasks", 3)
    current_load = min(estimated_active_hours / 8.0, 1.0)  # Normalize to 0-1

    return {
        "agent_id": agent_id,
        "name": agent_config.get("name", agent_id.upper()),
        "capabilities": agent_config.get("capabilities", []),
        "max_concurrent_tasks": max_tasks,
        "current_load": current_load,
        "is_theater": is_theater,
        "estimated_active_hours": estimated_active_hours,
        "cost_per_hour": agent_config.get("cost_per_hour_usd", 0.10),
    }


def capability_match_score(
    task_capabilities: List[str], agent_capabilities: List[str]
) -> float:
    """Score how well an agent's capabilities match a task's requirements.

    Args:
        task_capabilities: Required capabilities for the task.
        agent_capabilities: Agent's available capabilities.

    Returns:
        Match score from 0.0 (no match) to 1.0 (perfect match).
    """
    if not task_capabilities:
        return 0.5  # Any agent can do unspecified work

    matched = sum(1 for c in task_capabilities if c in agent_capabilities)
    return matched / len(task_capabilities)


def assign_tasks(
    tasks: List[Dict[str, Any]],
    agent_loads: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assign tasks to agents using heijunka (level-loading) strategy.

    Args:
        tasks: List of task dictionaries.
        agent_loads: Per-agent load information.

    Returns:
        List of assignment dictionaries with task_id, agent_id, and rationale.
    """
    # Sort tasks by priority (lower = higher priority), then deadline
    sorted_tasks = sorted(
        tasks,
        key=lambda t: (t.get("priority", 99), t.get("deadline", "9999-99-99")),
    )

    assignments: List[Dict[str, Any]] = []
    agent_task_counts: Dict[str, int] = {aid: 0 for aid in agent_loads}
    agent_hour_totals: Dict[str, float] = {aid: 0.0 for aid in agent_loads}

    for task in sorted_tasks:
        task_caps = task.get("required_capabilities", [])
        task_hours = task.get("estimated_hours", 2)
        best_agent: Optional[str] = None
        best_score: float = -1.0
        best_rationale: str = ""

        for aid, load_info in agent_loads.items():
            # Skip agents in theater loop
            if load_info["is_theater"]:
                continue

            # Capability match
            cap_score = capability_match_score(task_caps, load_info["capabilities"])
            if cap_score == 0:
                continue  # No capability match at all

            # Load balancing: prefer less-loaded agents
            current_tasks = agent_task_counts[aid]
            max_tasks = load_info["max_concurrent_tasks"]
            if current_tasks >= max_tasks:
                continue  # Agent at capacity

            load_score = 1.0 - (current_tasks / max_tasks)

            # Cost efficiency
            cost_score = 1.0 - min(load_info["cost_per_hour"] / 0.20, 1.0)

            # Combined score: capability match is most important
            total_score = (cap_score * 0.5) + (load_score * 0.35) + (cost_score * 0.15)

            if total_score > best_score:
                best_score = total_score
                best_agent = aid
                reasons = []
                if cap_score >= 0.5:
                    matched = [c for c in task_caps if c in load_info["capabilities"]]
                    reasons.append(f"capability match: {', '.join(matched)}")
                if load_score > 0.5:
                    reasons.append(f"low current load ({current_tasks}/{max_tasks} tasks)")
                if cost_score > 0.5:
                    reasons.append(f"cost-efficient (${load_info['cost_per_hour']}/hr)")
                best_rationale = "; ".join(reasons) if reasons else "best available"

        if best_agent:
            agent_task_counts[best_agent] += 1
            agent_hour_totals[best_agent] += task_hours
            assignments.append({
                "task_id": task["id"],
                "task_title": task["title"],
                "agent_id": best_agent,
                "agent_name": agent_loads[best_agent]["name"],
                "priority": task.get("priority", 99),
                "deadline": task.get("deadline", "TBD"),
                "estimated_hours": task_hours,
                "estimated_cost": task_hours * agent_loads[best_agent]["cost_per_hour"],
                "rationale": best_rationale,
                "capability_score": capability_match_score(
                    task_caps, agent_loads[best_agent]["capabilities"]
                ),
            })
        else:
            assignments.append({
                "task_id": task["id"],
                "task_title": task["title"],
                "agent_id": "UNASSIGNED",
                "agent_name": "UNASSIGNED",
                "priority": task.get("priority", 99),
                "deadline": task.get("deadline", "TBD"),
                "estimated_hours": task_hours,
                "estimated_cost": 0.0,
                "rationale": "No suitable agent found (capability mismatch or all at capacity)",
                "capability_score": 0.0,
            })

    return assignments


def generate_heijunka_board(
    assignments: List[Dict[str, Any]],
    agent_loads: Dict[str, Dict[str, Any]],
) -> str:
    """Generate HEIJUNKA_BOARD.md content.

    Args:
        assignments: Task assignment list.
        agent_loads: Per-agent load information.

    Returns:
        Markdown content for the heijunka board.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []

    lines.append("# 平準化 (Heijunka) Board — Level-Loaded Task Schedule")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Total Tasks:** {len(assignments)}")
    lines.append(
        f"**Assigned:** {sum(1 for a in assignments if a['agent_id'] != 'UNASSIGNED')}"
    )
    lines.append(
        f"**Unassigned:** {sum(1 for a in assignments if a['agent_id'] == 'UNASSIGNED')}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-agent assignments
    agents_with_tasks: Dict[str, List[Dict[str, Any]]] = {}
    for a in assignments:
        aid = a["agent_id"]
        if aid not in agents_with_tasks:
            agents_with_tasks[aid] = []
        agents_with_tasks[aid].append(a)

    for aid, tasks in agents_with_tasks.items():
        if aid == "UNASSIGNED":
            continue

        load = agent_loads.get(aid, {})
        total_hours = sum(t["estimated_hours"] for t in tasks)
        total_cost = sum(t["estimated_cost"] for t in tasks)

        status = "🔴 THEATER" if load.get("is_theater") else "🟢 ACTIVE"
        lines.append(f"## {load.get('name', aid)} ({status})")
        lines.append("")
        lines.append(f"**Assigned Tasks:** {len(tasks)} | **Total Hours:** {total_hours} | **Est. Cost:** {format_cost(total_cost)}")
        lines.append("")
        lines.append("| Priority | Task | Deadline | Hours | Rationale |")
        lines.append("|----------|------|----------|-------|-----------|")

        for t in sorted(tasks, key=lambda x: x["priority"]):
            p_marker = "🔴" if t["priority"] == 1 else "🟡" if t["priority"] == 2 else "🟢"
            lines.append(
                f"| {p_marker} P{t['priority']} | {t['task_title']} | "
                f"{t['deadline']} | {t['estimated_hours']}h | {t['rationale']} |"
            )

        lines.append("")

    # Unassigned tasks
    unassigned = agents_with_tasks.get("UNASSIGNED", [])
    if unassigned:
        lines.append("## ⚠️ UNASSIGNED Tasks")
        lines.append("")
        lines.append("| Priority | Task | Deadline | Hours | Reason |")
        lines.append("|----------|------|----------|-------|--------|")
        for t in unassigned:
            lines.append(
                f"| P{t['priority']} | {t['task_title']} | "
                f"{t['deadline']} | {t['estimated_hours']}h | {t['rationale']} |"
            )
        lines.append("")

    # Load distribution summary
    lines.append("---")
    lines.append("")
    lines.append("## Load Distribution")
    lines.append("")
    lines.append("| Agent | Tasks | Hours | Est. Cost | Load |")
    lines.append("|-------|-------|-------|-----------|------|")

    for aid, load in agent_loads.items():
        tasks = agents_with_tasks.get(aid, [])
        task_count = len(tasks)
        total_hours = sum(t["estimated_hours"] for t in tasks)
        total_cost = sum(t["estimated_cost"] for t in tasks)
        load_bar_filled = int(min(task_count / load["max_concurrent_tasks"], 1.0) * 5)
        load_bar = "█" * load_bar_filled + "░" * (5 - load_bar_filled)
        theater_marker = " ⛔" if load["is_theater"] else ""

        lines.append(
            f"| {load['name']}{theater_marker} | {task_count}/{load['max_concurrent_tasks']} | "
            f"{total_hours}h | {format_cost(total_cost)} | {load_bar} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*平準化 — Level the load. Balance the flow. Optimize the whole.*")

    return "\n".join(lines)


def generate_intervention(
    agent_id: str,
    agent_name: str,
    tasks: List[Dict[str, Any]],
) -> str:
    """Generate INTERVENTION.md content for a specific agent.

    Args:
        agent_id: Agent identifier.
        agent_name: Display name of the agent.
        tasks: Tasks assigned to this agent.

    Returns:
        Markdown content for the agent's intervention file.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []

    lines.append(f"# INTERVENTION — {agent_name}")
    lines.append("")
    lines.append(f"**Issued:** {now}")
    lines.append(f"**From:** META_META_KNOWER Heijunka Scheduler")
    lines.append(f"**To:** {agent_name} ({agent_id})")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Assigned Tasks (Priority Order)")
    lines.append("")

    for i, t in enumerate(sorted(tasks, key=lambda x: x["priority"]), 1):
        lines.append(f"### {i}. [{t['task_id']}] {t['task_title']}")
        lines.append(f"- **Priority:** P{t['priority']}")
        lines.append(f"- **Deadline:** {t['deadline']}")
        lines.append(f"- **Estimated Hours:** {t['estimated_hours']}")
        lines.append(f"- **Why You:** {t['rationale']}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Execution Protocol")
    lines.append("")
    lines.append("1. Complete tasks in priority order (P1 first)")
    lines.append("2. Log each deliverable in JIKOKU format")
    lines.append("3. If blocked, report immediately — do not spin in theater")
    lines.append("4. One piece flow: finish one task before starting the next")
    lines.append("")
    lines.append("*一個流し — Ship ONE thing, then the next.*")

    return "\n".join(lines)


@click.command()
@click.option("--tasks", "tasks_file", default=None, help="YAML file with pending tasks")
@click.option("--dry-run", is_flag=True, help="Print assignments without writing files")
@click.option("--output-dir", default=None, help="Directory for output files")
def main(
    tasks_file: Optional[str],
    dry_run: bool,
    output_dir: Optional[str],
) -> None:
    """Run heijunka level-loaded task scheduling."""
    config = load_config()

    # Load tasks
    if tasks_file:
        tasks_path = Path(tasks_file)
        if not tasks_path.exists():
            click.echo(f"❌ Tasks file not found: {tasks_path}")
            return
        with open(tasks_path) as f:
            task_data = yaml.safe_load(f)
            tasks = task_data.get("tasks", task_data) if isinstance(task_data, dict) else task_data
    else:
        click.echo("📋 No tasks file specified, using default sample tasks")
        tasks = DEFAULT_TASKS

    # Compute agent loads
    entries = load_audit_log(days=7)
    agent_ids = list(config.get("agents", {}).keys())
    agent_loads = {
        aid: compute_agent_load(entries, aid, config) for aid in agent_ids
    }

    click.echo(f"🏭 Scheduling {len(tasks)} tasks across {len(agent_ids)} agents...")
    click.echo("")

    # Theater status report
    for aid, load in agent_loads.items():
        status = "🔴 THEATER" if load["is_theater"] else "🟢 AVAILABLE"
        click.echo(
            f"  {load['name']:>8s}: {status} "
            f"(load: {load['current_load']:.0%}, "
            f"caps: {', '.join(load['capabilities'][:3])}...)"
        )
    click.echo("")

    # Assign tasks
    assignments = assign_tasks(tasks, agent_loads)

    # Print summary
    assigned = [a for a in assignments if a["agent_id"] != "UNASSIGNED"]
    unassigned = [a for a in assignments if a["agent_id"] == "UNASSIGNED"]

    click.echo(f"✅ Assigned: {len(assigned)}/{len(assignments)} tasks")
    if unassigned:
        click.echo(f"⚠️  Unassigned: {len(unassigned)} tasks")
        for u in unassigned:
            click.echo(f"   - {u['task_title']}: {u['rationale']}")
    click.echo("")

    for a in assigned:
        click.echo(
            f"  [{a['task_id']}] {a['task_title']} → {a['agent_name']} "
            f"(P{a['priority']}, {a['estimated_hours']}h, "
            f"{format_cost(a['estimated_cost'])})"
        )
    click.echo("")

    if dry_run:
        click.echo("🏁 Dry run complete. No files written.")
        return

    # Write output files
    out_dir = Path(output_dir) if output_dir else REPO_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    # HEIJUNKA_BOARD.md
    board_content = generate_heijunka_board(assignments, agent_loads)
    board_path = out_dir / "HEIJUNKA_BOARD.md"
    board_path.write_text(board_content)
    click.echo(f"📋 Heijunka board: {board_path}")

    # Per-agent INTERVENTION.md files
    agent_tasks: Dict[str, List[Dict[str, Any]]] = {}
    for a in assigned:
        aid = a["agent_id"]
        if aid not in agent_tasks:
            agent_tasks[aid] = []
        agent_tasks[aid].append(a)

    for aid, at in agent_tasks.items():
        agent_name = agent_loads[aid]["name"]
        intervention = generate_intervention(aid, agent_name, at)
        intervention_path = out_dir / f"INTERVENTION_{aid.upper()}.md"
        intervention_path.write_text(intervention)
        click.echo(f"📝 Intervention for {agent_name}: {intervention_path}")

    # Save machine-readable assignments
    assignments_path = out_dir / "heijunka_assignments.json"
    assignments_path.write_text(json.dumps(assignments, indent=2, default=str))
    click.echo(f"💾 Assignments JSON: {assignments_path}")

    click.echo("")
    click.echo("🏭 平準化 complete. Tasks level-loaded across available agents.")


if __name__ == "__main__":
    main()
