#!/usr/bin/env python3
"""
MCP Monitor CLI — Direct Interface for Inter-Agent Communication

This CLI allows direct interaction without MCP protocol overhead.
Useful for testing and manual monitoring.

Usage:
    # Post checkpoint (OpenClawd)
    python3 -m mcp_monitor.cli checkpoint --model mixtral-8x7b --progress "25/50 pairs" --d -2.1 --p 0.001 --mem 68.2
    
    # Get checkpoints (Cursor)
    python3 -m mcp_monitor.cli checkpoints --limit 5
    
    # Post finding
    python3 -m mcp_monitor.cli finding --source cursor --type suggestion --content "Increase sample size"
    
    # Get status
    python3 -m mcp_monitor.cli status
    
    # Verify logging
    python3 -m mcp_monitor.cli verify --path results/canonical/...
"""

import argparse
import json
from pathlib import Path

try:
    from .server import MCPServer
except ImportError:
    from server import MCPServer


def main():
    parser = argparse.ArgumentParser(description="MCP Monitor CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # checkpoint command
    cp = subparsers.add_parser("checkpoint", help="Post a checkpoint")
    cp.add_argument("--model", required=True)
    cp.add_argument("--progress", required=True)
    cp.add_argument("--d", type=float, required=True, help="Partial Cohen's d")
    cp.add_argument("--p", type=float, required=True, help="Partial p-value")
    cp.add_argument("--mem", type=float, required=True, help="GPU memory GB")
    cp.add_argument("--anomalies", nargs="*", default=[])
    
    # checkpoints command
    cps = subparsers.add_parser("checkpoints", help="Get checkpoints")
    cps.add_argument("--limit", type=int, default=10)
    cps.add_argument("--since", type=str)
    
    # finding command
    f = subparsers.add_parser("finding", help="Post a finding")
    f.add_argument("--source", required=True, choices=["cursor", "openclawd"])
    f.add_argument("--type", required=True, choices=["result", "insight", "concern", "suggestion"])
    f.add_argument("--content", required=True)
    f.add_argument("--evidence", type=str)
    f.add_argument("--priority", default="medium", choices=["low", "medium", "high", "critical"])
    
    # findings command
    fs = subparsers.add_parser("findings", help="Get findings")
    fs.add_argument("--source", type=str)
    fs.add_argument("--type", type=str)
    fs.add_argument("--unacknowledged", action="store_true")
    
    # suggest command
    s = subparsers.add_parser("suggest", help="Suggest experiment")
    s.add_argument("--experiment", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--rationale", required=True)
    s.add_argument("--priority", type=int, required=True)
    s.add_argument("--config", type=str)
    
    # status command
    st = subparsers.add_parser("status", help="Get experiment status")
    
    # start command
    start = subparsers.add_parser("start", help="Start experiment")
    start.add_argument("--experiment", required=True)
    start.add_argument("--model", required=True)
    start.add_argument("--config", required=True)
    
    # end command
    end = subparsers.add_parser("end", help="End experiment")
    end.add_argument("--results", required=True)
    end.add_argument("--success", action="store_true")
    end.add_argument("--failed", action="store_true")
    
    # verify command
    v = subparsers.add_parser("verify", help="Verify logging")
    v.add_argument("--path", required=True)
    
    args = parser.parse_args()
    server = MCPServer()
    
    if args.command == "checkpoint":
        result = server.post_checkpoint(
            model=args.model,
            progress=args.progress,
            partial_d=args.d,
            partial_p=args.p,
            gpu_memory_gb=args.mem,
            anomalies=args.anomalies
        )
    
    elif args.command == "checkpoints":
        result = server.get_checkpoints(limit=args.limit, since=args.since)
    
    elif args.command == "finding":
        result = server.post_finding(
            source=args.source,
            finding_type=args.type,
            content=args.content,
            evidence=args.evidence,
            priority=args.priority
        )
    
    elif args.command == "findings":
        result = server.get_findings(
            source=args.source,
            finding_type=args.type,
            unacknowledged_only=args.unacknowledged
        )
    
    elif args.command == "suggest":
        result = server.suggest_experiment(
            experiment=args.experiment,
            model=args.model,
            rationale=args.rationale,
            priority=args.priority,
            config_path=args.config
        )
    
    elif args.command == "status":
        result = server.get_experiment_status()
    
    elif args.command == "start":
        result = server.start_experiment(
            experiment=args.experiment,
            model=args.model,
            config_path=args.config
        )
    
    elif args.command == "end":
        success = args.success or not args.failed
        result = server.end_experiment(
            results_path=args.results,
            success=success
        )
    
    elif args.command == "verify":
        result = server.verify_logging(results_path=args.path)
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
