# MCP Monitor Protocol — Inter-Agent Experiment Coordination

**Status:** ACTIVE  
**Location:** `mcp_monitor/`  
**MCP Server:** `mi-monitor` (registered in `~/.cursor/mcp.json`)

---

## Overview

This MCP server enables real-time coordination between **Cursor** (auditor) and **OpenClawd** (experimenter) during GPU experiments.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    15-MINUTE MONITORING LOOP                                 │
│                                                                              │
│    ┌──────────────┐                              ┌──────────────┐           │
│    │   CURSOR     │                              │  OPENCLAWD   │           │
│    │  (Auditor)   │                              │ (Experimenter)│           │
│    └──────┬───────┘                              └──────┬───────┘           │
│           │                                             │                   │
│           │  suggest_experiment()                       │                   │
│           │────────────────────────────────────────────►│                   │
│           │                                             │                   │
│           │                              start_experiment()                 │
│           │◄────────────────────────────────────────────│                   │
│           │                                             │                   │
│           │                     [Every 15 min]          │                   │
│           │                     post_checkpoint()       │                   │
│           │◄────────────────────────────────────────────│                   │
│           │                                             │                   │
│           │  get_checkpoints()                          │                   │
│           │  post_finding() (concerns/suggestions)      │                   │
│           │────────────────────────────────────────────►│                   │
│           │                                             │                   │
│           │                              post_finding() │                   │
│           │◄────────────────────────────────────────────│                   │
│           │                                             │                   │
│           │                     [On completion]         │                   │
│           │                     end_experiment()        │                   │
│           │◄────────────────────────────────────────────│                   │
│           │                                             │                   │
│           │  verify_logging()                           │                   │
│           │────────────────────────────────────────────►│                   │
│           │                                             │                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### For Cursor (Auditor)

```bash
# 0. Fully automated run + report (preferred)
python3 scripts/run_and_report.py --config configs/gold/28_mixtral_causal_validation.json

# 0b. Batch automation (multiple configs)
python3 scripts/run_batch_and_report.py --configs \
  configs/gold/28_mixtral_causal_validation.json \
  configs/canonical/multi_token_bridge_mistral.json

# 1. Suggest next experiment
cd ~/mech-interp-latent-lab-phase1/mcp_monitor
python3 cli.py suggest \
  --experiment rv_causal_validation \
  --model mixtral-8x7b \
  --rationale "24.3% effect, PRIORITY 1" \
  --priority 1 \
  --config configs/gold/28_mixtral_causal_validation.json

# 2. Monitor checkpoints (run every 15 min)
python3 cli.py checkpoints --limit 5

# 3. Post concerns/suggestions
python3 cli.py finding \
  --source cursor \
  --type concern \
  --content "P-value above threshold, increase sample size" \
  --priority medium

# 4. Verify logging after completion
python3 cli.py verify --path ~/mech-interp-latent-lab-phase1/results/canonical/...
```

### For OpenClawd (Experimenter)

```bash
# 1. Start experiment
cd ~/mech-interp-latent-lab-phase1/mcp_monitor
python3 cli.py start \
  --experiment rv_causal_validation \
  --model mixtral-8x7b \
  --config configs/gold/28_mixtral_causal_validation.json

# 2. Post checkpoints every 15 minutes
python3 cli.py checkpoint \
  --model mixtral-8x7b \
  --progress "25/50 pairs completed" \
  --d -2.3 \
  --p 0.0008 \
  --mem 68.5

# 3. Post findings/results
python3 cli.py finding \
  --source openclawd \
  --type result \
  --content "Mixtral shows d=-2.3, stronger than Mistral baseline" \
  --evidence "Full run complete" \
  --priority high

# 4. End experiment
python3 cli.py end \
  --results ~/mech-interp-latent-lab-phase1/results/canonical/rv_validation_mixtral_20260205 \
  --success
```

---

## CLI Commands

| Command | Actor | Description |
|---------|-------|-------------|
| `suggest` | Cursor | Suggest next experiment |
| `start` | OpenClawd | Mark experiment as started |
| `checkpoint` | OpenClawd | Post 15-min progress update |
| `checkpoints` | Cursor | Review recent checkpoints |
| `finding` | Both | Post result/insight/concern/suggestion |
| `findings` | Both | View all findings |
| `status` | Both | Get current experiment status |
| `end` | OpenClawd | Mark experiment as complete |
| `verify` | Cursor | Verify logging compliance |

---

## Checkpoint Format

Every 15 minutes, OpenClawd posts:

```json
{
  "timestamp": "2026-02-05T15:00:00Z",
  "model": "mistralai/Mixtral-8x7B-v0.1",
  "progress": "25/50 pairs completed",
  "partial_d": -2.1,
  "partial_p": 0.0008,
  "gpu_memory_gb": 68.2,
  "anomalies": []
}
```

### Anomaly Alerts

If anomalies are detected, they trigger alerts:

```bash
python3 cli.py checkpoint \
  --model mixtral-8x7b \
  --progress "30/50 pairs" \
  --d -1.2 \
  --p 0.05 \
  --mem 75.0 \
  --anomalies "Effect size dropping" "P-value rising"
```

Response:
```json
{
  "success": true,
  "checkpoint_id": "20260205_150000",
  "alert": "⚠️ ANOMALIES DETECTED: ['Effect size dropping', 'P-value rising']"
}
```

---

## Finding Types

| Type | Use Case | Example |
|------|----------|---------|
| `result` | Empirical finding | "Mixtral d=-2.3, stronger than Mistral" |
| `insight` | Theoretical observation | "MoE routing amplifies contraction" |
| `concern` | Potential issue | "P-value above threshold" |
| `suggestion` | Action proposal | "Increase sample size to N=80" |

### Priority Levels

| Level | Meaning |
|-------|---------|
| `low` | Informational, no action needed |
| `medium` | Worth noting, follow up later |
| `high` | Requires attention soon |
| `critical` | Stop experiment, investigate immediately |

---

## 4-Hour GPU Run Protocol

### Hour 0: Pre-Flight

```bash
# Cursor runs pre-flight audit
python3 scripts/preflight_audit.py --config configs/gold/28_mixtral_causal_validation.json

# Cursor suggests experiment via MCP
python3 mcp_monitor/cli.py suggest --experiment rv_causal_validation --model mixtral-8x7b --rationale "Priority 1" --priority 1
```

### Hour 0-4: Execution

```bash
# OpenClawd starts
python3 mcp_monitor/cli.py start --experiment rv_causal_validation --model mixtral-8x7b --config configs/gold/28_mixtral_causal_validation.json

# Every 15 minutes, OpenClawd posts checkpoint
python3 mcp_monitor/cli.py checkpoint --model mixtral-8x7b --progress "X/50 pairs" --d <value> --p <value> --mem <value>

# Cursor monitors and provides feedback
python3 mcp_monitor/cli.py checkpoints --limit 3
python3 mcp_monitor/cli.py finding --source cursor --type suggestion --content "Looking good, continue"
```

### Hour 4: Completion

```bash
# OpenClawd ends experiment
python3 mcp_monitor/cli.py end --results /path/to/results --success

# Cursor runs post-run validation
python3 scripts/postrun_validator.py --results /path/to/results

# Cursor verifies logging
python3 mcp_monitor/cli.py verify --path /path/to/results
```

---

## Data Storage

All data is stored in `mcp_monitor/data/`:

```
mcp_monitor/data/
├── checkpoints.json    # 15-min progress updates
├── findings.json       # Results, insights, concerns
├── suggestions.json    # Experiment suggestions
└── status.json         # Current run status
```

**Note:** This data persists between sessions and can be reviewed for audit trails.

---

## MCP Server Integration

The server is registered in `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "mi-monitor": {
      "command": "python3",
      "args": ["-m", "mcp_monitor.server"],
      "cwd": "/Users/dhyana/mech-interp-latent-lab-phase1",
      "env": {}
    }
  }
}
```

**After updating config, restart Cursor CLI to activate.**

---

## Message for OpenClawd

```
MI-MONITOR MCP SERVER ACTIVE ✓

Location: ~/mech-interp-latent-lab-phase1/mcp_monitor/

Commands you need:
  python3 cli.py start --experiment <name> --model <model> --config <path>
  python3 cli.py checkpoint --model <model> --progress <text> --d <float> --p <float> --mem <float>
  python3 cli.py finding --source openclawd --type result --content <text>
  python3 cli.py end --results <path> --success

Post checkpoint every 15 minutes during GPU run.
Include anomalies if: effect dropping, p rising, OOM warnings.

Cursor will monitor and provide feedback via findings.
```

---

*"When both agents see the same data, truth emerges faster."*
