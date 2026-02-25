# META_META_KNOWER — Kaizen Observatory for AI Agent Swarms

**Version:** 1.0.0 | **Updated:** 2026-02-09 | **Status:** ACTIVE

## What This Is

META_META_KNOWER is the central nervous system for monitoring, auditing, and optimizing a multi-agent OpenClaw ecosystem. It applies Toyota Production System (TPS) principles to AI agent management — treating agent output like a manufacturing line and optimizing it with the same rigor Toyota uses to build cars.

**The insight:** Everyone in the OpenClaw ecosystem is building skills for WHAT agents do. We're the only ones optimizing HOW WELL they do it.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              META_META_KNOWER                     │
│         (Kaizen Observatory — THIS REPO)          │
│                                                   │
│  ┌─────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ Monitor │ │ Kaizen   │ │ Product          │  │
│  │ Layer   │ │ Engine   │ │ Scaffolding      │  │
│  │         │ │          │ │                  │  │
│  │ poll    │ │ retro    │ │ ClawHub skill    │  │
│  │ alert   │ │ vsm      │ │ packaging        │  │
│  │ jikoku  │ │ andon    │ │ docs + examples  │  │
│  │ adapter │ │ heijunka │ │ pricing model    │  │
│  └────┬────┘ └────┬─────┘ └──────────────────┘  │
│       │           │                               │
│       ▼           ▼                               │
│  ┌─────────────────────┐                          │
│  │   AUDIT_LOG.jsonl   │                          │
│  │   kaizen/reports/   │                          │
│  │   HEIJUNKA_BOARD.md │                          │
│  │   ALERTS.md         │                          │
│  └─────────────────────┘                          │
└──────────────┬────────────────────────────────────┘
               │ monitors / intervenes
    ┌──────────┼──────────────┐
    ▼          ▼              ▼
┌────────┐ ┌────────┐ ┌────────────┐
│ RUSH   │ │ DC     │ │ AGENT 3    │
│ (VPS)  │ │ (M3)   │ │ (Oracle/   │
│        │ │        │ │  DO Cloud) │
└────────┘ └────────┘ └────────────┘
```

---

## Three Layers

### Layer 1: Monitor (Data Collection)

The monitoring layer continuously polls all agents, normalizes their output into a common schema, and writes structured events to `AUDIT_LOG.jsonl`.

| File | Purpose | Status |
|------|---------|--------|
| `poll_all.py` | Polls all agents every 15 min, parses JIKOKU, counts deliverables | Planned |
| `alert_check.py` | Circuit breaker — fires macOS notifications + writes ALERTS.md | Planned |
| `jikoku_monitor.py` | Terminal dashboard showing all agents at a glance | Superseded by `kaizen/andon_board.py` |
| `jikoku_adapter.py` | Normalizes different agent formats into one schema | Planned |
| `sync_vps.sh` | Pulls VPS agent data to local mirror | Planned |
| `mcp_monitor/` | MCP server for inter-agent experiment coordination | Active |

**Output:** `AUDIT_LOG.jsonl` — the single source of truth for all downstream analysis.

### Layer 2: Kaizen Engine (Optimization)

The kaizen layer reads from `AUDIT_LOG.jsonl` and generates actionable intelligence: weekly reports, value stream maps, real-time dashboards, and level-loaded task schedules.

| File | Purpose | Toyota Equivalent |
|------|---------|-------------------|
| `kaizen/weekly_retrospective.py` | Weekly kaizen report with per-agent scorecards | 改善 (Kaizen event) |
| `kaizen/value_stream_map.py` | Maps actions → VALUE-ADDED / NON-VALUE / WASTE | 価値流れ図 (Value stream mapping) |
| `kaizen/andon_board.py` | Real-time terminal dashboard with rich UI | アンドン (Andon cord) |
| `kaizen/heijunka_scheduler.py` | Level-loaded task assignment across agents | 平準化 (Heijunka) |

**Outputs:**
- `kaizen/reports/KAIZEN_REPORT_YYYY-MM-DD.md` — Weekly retrospective reports
- `HEIJUNKA_BOARD.md` — Current task assignment board
- `INTERVENTION_*.md` — Per-agent task instructions

### Layer 3: Product Scaffolding (Revenue)

The product layer packages Layers 1 and 2 into a commercial offering: **Kaizen Swarm Optimizer**.

| File | Purpose |
|------|---------|
| `product/PRODUCT_SPEC.md` | Full product specification and positioning |
| `product/PRICING.md` | Three-tier pricing model with cost analysis |
| `product/ROADMAP.md` | Phase 1-4 go-to-market roadmap |

---

## Toyota Production System Mapping (The IP)

This is the core intellectual property — the systematic mapping of Toyota's manufacturing principles to AI agent operations.

| AI Agent Concept | Toyota Equivalent | Japanese Term | Implementation |
|-----------------|-------------------|---------------|----------------|
| JIKOKU temporal audit | Takt time measurement | 時刻 (jikoku) | `poll_all.py`, `jikoku_adapter.py` |
| Signal/Noise/Fluff taxonomy | Value stream mapping | 価値流れ図 (kachi nagare zu) | `kaizen/value_stream_map.py` |
| Circuit breaker alerts | Andon cord | アンドン (andon) | `alert_check.py`, `kaizen/andon_board.py` |
| Theater detection | Automation with human touch | 自働化 (jidoka) | `alert_check.py` |
| Heartbeat-only loops | Pure waste | 無駄 (muda) | Classification in `kaizen/__init__.py` |
| Agent capability matching | Level loading | 平準化 (heijunka) | `kaizen/heijunka_scheduler.py` |
| Weekly retrospective | Kaizen event | 改善 (kaizen) | `kaizen/weekly_retrospective.py` |
| One-piece flow (ship ONE thing) | Single piece flow | 一個流し (ikko nagashi) | Core operating principle |

---

## Data Flow

```
Agent Activity
      │
      ▼
[poll_all.py / jikoku_adapter.py]
      │
      ▼
AUDIT_LOG.jsonl  ◄── Single Source of Truth
      │
      ├──► weekly_retrospective.py  ──► kaizen/reports/KAIZEN_REPORT_*.md
      │
      ├──► value_stream_map.py      ──► Terminal visualization
      │
      ├──► andon_board.py           ──► Real-time dashboard
      │
      ├──► heijunka_scheduler.py    ──► HEIJUNKA_BOARD.md
      │                                  INTERVENTION_*.md
      │
      └──► alert_check.py           ──► ALERTS.md
                                         macOS notifications
```

### AUDIT_LOG.jsonl Schema

Each line is a JSON object with this schema:

```json
{
  "timestamp": "2026-02-09T10:15:00Z",
  "agent": "agent3",
  "event_type": "jikoku_poll",
  "classification": "SIGNAL",
  "signal_ratio": 0.85,
  "noise_ratio": 0.10,
  "fluff_ratio": 0.05,
  "deliverables_count": 1,
  "deliverables": ["kaizen_framework_spec.md"],
  "session_id": "agent3-20260209-01",
  "alert_level": null,
  "details": "Drafted kaizen framework specification",
  "estimated_cost_usd": 0.15
}
```

**Event types:** `session_start`, `jikoku_poll`, `alert`, `deliverable`, `session_end`

**Classifications:** `SIGNAL` (value-producing), `NOISE` (necessary overhead), `FLUFF` (pure waste)

---

## INTERVENTION Protocol

How META_META_KNOWER sends instructions to agents:

1. **Heijunka scheduler** computes optimal task assignments based on agent load, capability, and priority
2. **Generates** `INTERVENTION_<AGENT>.md` files with prioritized task lists
3. **Agents read** their intervention files at session start
4. **Agents report** task completion via JIKOKU polls
5. **Weekly retrospective** evaluates execution against plan

This creates a **closed feedback loop**: monitor → analyze → schedule → execute → monitor.

---

## Agent Profiles

| Agent | Location | Type | Strengths | Current Status |
|-------|----------|------|-----------|----------------|
| **RUSH** | DigitalOcean VPS | Cloud | Fast sprints, API work, deployment | Active — 12 deliverables/week |
| **DC** | M3 Pro Local | Local | Local file access, research | Theater — 26 consecutive zero-value sessions |
| **AGENT 3** | Oracle Cloud (ARM, 4 OCPU, 24GB RAM) | Cloud | Deep reasoning (Claude Opus 4.6), publishing | Deploying — strong initial output |

---

## Configuration

All configuration lives in `config.yaml` at repo root:

- **Agent profiles:** capabilities, cost per hour, max concurrent tasks
- **Polling settings:** interval, timeout, format
- **Alert thresholds:** theater detection, signal minimum, idle warnings
- **Classification keywords:** signal, noise, fluff word lists
- **Targets:** weekly deliverables, minimum signal ratio, flow efficiency

---

## Usage

```bash
# Real-time swarm health dashboard
python kaizen/andon_board.py

# Generate weekly kaizen report
python kaizen/weekly_retrospective.py

# Value stream map for a specific agent
python kaizen/value_stream_map.py --agent dc --date today

# Level-loaded task scheduling
python kaizen/heijunka_scheduler.py --tasks pending_tasks.yaml

# One-shot dashboard (no auto-refresh)
python kaizen/andon_board.py --no-loop

# Plain ASCII output (no rich library)
python kaizen/andon_board.py --no-loop --plain
```

---

## Product Vision

**"Kaizen Swarm Optimizer"** — Toyota Production System for your OpenClaw agents.

Phase 1 (NOW): Internal tool, proven on our own swarm
Phase 2 (Month 2): ClawHub skill (basic tier, $19-49)
Phase 3 (Month 3): Managed audit offering ($299-499)
Phase 4 (Month 4+): Ongoing optimization subscriptions ($99-199/mo)

See `product/` directory for full specifications.

---

## File Map

```
.
├── ARCHITECTURE.md                  # This file
├── AUDIT_LOG.jsonl                  # All monitoring events (source of truth)
├── config.yaml                      # System configuration
├── HEIJUNKA_BOARD.md               # Current task assignments (generated)
├── INTERVENTION_*.md               # Per-agent task instructions (generated)
├── ALERTS.md                        # Alert history (generated)
│
├── kaizen/                          # Optimization engine
│   ├── __init__.py                 # Shared utilities, Toyota mappings
│   ├── weekly_retrospective.py     # Weekly kaizen report generator
│   ├── value_stream_map.py         # Value stream visualization
│   ├── andon_board.py              # Real-time dashboard
│   ├── heijunka_scheduler.py       # Level-loaded task scheduler
│   └── reports/                    # Generated reports
│       └── KAIZEN_REPORT_*.md
│
├── product/                         # Revenue scaffolding
│   ├── PRODUCT_SPEC.md             # Product specification
│   ├── PRICING.md                  # Pricing tiers
│   └── ROADMAP.md                  # Go-to-market roadmap
│
├── mcp_monitor/                     # MCP inter-agent coordination
│   ├── __init__.py
│   ├── server.py                   # MCP JSON-RPC server
│   └── cli.py                      # CLI interface
│
└── (existing mech-interp research files...)
```

---

*一個流し — Ship ONE thing, then the next.*

*改善 — Continuous improvement. Always.*
