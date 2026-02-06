#!/usr/bin/env python3
"""
MCP Monitor Server — Inter-Agent Experiment Coordination

This MCP server enables Cursor to:
1. Monitor experiment progress every 15 minutes
2. Exchange findings with OpenClawd
3. Suggest next experiments
4. Verify logging compliance

Tools:
- post_checkpoint: OpenClawd posts progress updates
- get_checkpoints: Cursor retrieves all checkpoints
- post_finding: Either agent posts a finding
- get_findings: Retrieve all findings
- suggest_experiment: Cursor suggests next experiment
- verify_logging: Verify artifacts are properly logged
- get_experiment_status: Get current run status

Usage:
    python3 -m mcp_monitor.server
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# MCP protocol implementation
class MCPServer:
    def __init__(self):
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Data files
        self.checkpoints_file = self.data_dir / "checkpoints.json"
        self.findings_file = self.data_dir / "findings.json"
        self.suggestions_file = self.data_dir / "suggestions.json"
        self.status_file = self.data_dir / "status.json"
        
        # Initialize files
        for f in [self.checkpoints_file, self.findings_file, self.suggestions_file]:
            if not f.exists():
                f.write_text("[]")
        if not self.status_file.exists():
            self.status_file.write_text(json.dumps({
                "current_experiment": None,
                "started_at": None,
                "last_checkpoint": None,
                "status": "idle"
            }))
    
    def _load_json(self, path: Path) -> Any:
        return json.loads(path.read_text())
    
    def _save_json(self, path: Path, data: Any):
        path.write_text(json.dumps(data, indent=2, default=str))
    
    # Tool implementations
    def post_checkpoint(self, 
                       model: str,
                       progress: str,
                       partial_d: float,
                       partial_p: float,
                       gpu_memory_gb: float,
                       anomalies: List[str] = None) -> Dict:
        """Post a 15-minute checkpoint from OpenClawd."""
        checkpoint = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "progress": progress,
            "partial_d": partial_d,
            "partial_p": partial_p,
            "gpu_memory_gb": gpu_memory_gb,
            "anomalies": anomalies or [],
            "source": "openclawd"
        }
        
        checkpoints = self._load_json(self.checkpoints_file)
        checkpoints.append(checkpoint)
        self._save_json(self.checkpoints_file, checkpoints)
        
        # Update status
        status = self._load_json(self.status_file)
        status["last_checkpoint"] = checkpoint["timestamp"]
        self._save_json(self.status_file, status)
        
        # Alert if anomalies
        alert = None
        if anomalies:
            alert = f"⚠️ ANOMALIES DETECTED: {anomalies}"
        
        return {
            "success": True,
            "checkpoint_id": checkpoint["id"],
            "alert": alert
        }
    
    def get_checkpoints(self, limit: int = 10, since: str = None) -> Dict:
        """Get recent checkpoints (for Cursor to review)."""
        checkpoints = self._load_json(self.checkpoints_file)
        
        if since:
            checkpoints = [c for c in checkpoints if c["timestamp"] > since]
        
        return {
            "checkpoints": checkpoints[-limit:],
            "total": len(checkpoints)
        }
    
    def post_finding(self,
                    source: str,  # "cursor" or "openclawd"
                    finding_type: str,  # "result", "insight", "concern", "suggestion"
                    content: str,
                    evidence: str = None,
                    priority: str = "medium") -> Dict:
        """Post a finding from either agent."""
        finding = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "type": finding_type,
            "content": content,
            "evidence": evidence,
            "priority": priority,
            "acknowledged": False
        }
        
        findings = self._load_json(self.findings_file)
        findings.append(finding)
        self._save_json(self.findings_file, findings)
        
        return {
            "success": True,
            "finding_id": finding["id"]
        }
    
    def get_findings(self, 
                    source: str = None, 
                    finding_type: str = None,
                    unacknowledged_only: bool = False) -> Dict:
        """Get findings, optionally filtered."""
        findings = self._load_json(self.findings_file)
        
        if source:
            findings = [f for f in findings if f["source"] == source]
        if finding_type:
            findings = [f for f in findings if f["type"] == finding_type]
        if unacknowledged_only:
            findings = [f for f in findings if not f["acknowledged"]]
        
        return {
            "findings": findings,
            "total": len(findings)
        }
    
    def acknowledge_finding(self, finding_id: str) -> Dict:
        """Mark a finding as acknowledged."""
        findings = self._load_json(self.findings_file)
        
        for f in findings:
            if f["id"] == finding_id:
                f["acknowledged"] = True
                f["acknowledged_at"] = datetime.now().isoformat()
                break
        
        self._save_json(self.findings_file, findings)
        return {"success": True}
    
    def suggest_experiment(self,
                          experiment: str,
                          model: str,
                          rationale: str,
                          priority: int,
                          config_path: str = None) -> Dict:
        """Cursor suggests next experiment to OpenClawd."""
        suggestion = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "experiment": experiment,
            "model": model,
            "rationale": rationale,
            "priority": priority,
            "config_path": config_path,
            "status": "pending"
        }
        
        suggestions = self._load_json(self.suggestions_file)
        suggestions.append(suggestion)
        self._save_json(self.suggestions_file, suggestions)
        
        return {
            "success": True,
            "suggestion_id": suggestion["id"]
        }
    
    def get_suggestions(self, status: str = None) -> Dict:
        """Get experiment suggestions."""
        suggestions = self._load_json(self.suggestions_file)
        
        if status:
            suggestions = [s for s in suggestions if s["status"] == status]
        
        return {
            "suggestions": suggestions,
            "total": len(suggestions)
        }
    
    def update_suggestion_status(self, suggestion_id: str, status: str) -> Dict:
        """Update suggestion status (pending, accepted, rejected, completed)."""
        suggestions = self._load_json(self.suggestions_file)
        
        for s in suggestions:
            if s["id"] == suggestion_id:
                s["status"] = status
                s["updated_at"] = datetime.now().isoformat()
                break
        
        self._save_json(self.suggestions_file, suggestions)
        return {"success": True}
    
    def start_experiment(self, 
                        experiment: str,
                        model: str,
                        config_path: str) -> Dict:
        """Mark experiment as started."""
        status = {
            "current_experiment": experiment,
            "model": model,
            "config_path": config_path,
            "started_at": datetime.now().isoformat(),
            "last_checkpoint": None,
            "status": "running"
        }
        self._save_json(self.status_file, status)
        
        return {"success": True, "status": status}
    
    def end_experiment(self, 
                      results_path: str,
                      success: bool,
                      summary: Dict = None) -> Dict:
        """Mark experiment as ended."""
        status = self._load_json(self.status_file)
        status["status"] = "completed" if success else "failed"
        status["ended_at"] = datetime.now().isoformat()
        status["results_path"] = results_path
        status["summary"] = summary
        self._save_json(self.status_file, status)
        
        return {"success": True, "status": status}
    
    def get_experiment_status(self) -> Dict:
        """Get current experiment status."""
        return self._load_json(self.status_file)
    
    def verify_logging(self, results_path: str) -> Dict:
        """Verify artifacts are properly logged."""
        path = Path(results_path).expanduser()
        
        required = ["config.json", "summary.json"]
        recommended = ["per_sample.csv", "hardware_info.json"]
        
        issues = []
        for f in required:
            if not (path / f).exists():
                issues.append(f"MISSING REQUIRED: {f}")
        
        warnings = []
        for f in recommended:
            if not (path / f).exists():
                warnings.append(f"MISSING RECOMMENDED: {f}")
        
        # Check summary.json content
        summary_path = path / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            required_fields = ["cohens_d", "p_value", "n_pairs", "controls_passed"]
            for field in required_fields:
                if field not in summary:
                    issues.append(f"MISSING FIELD in summary.json: {field}")
        
        valid = len(issues) == 0
        
        return {
            "valid": valid,
            "issues": issues,
            "warnings": warnings,
            "path": str(path)
        }
    
    def get_tools(self) -> List[Dict]:
        """Return MCP tool definitions."""
        return [
            {
                "name": "post_checkpoint",
                "description": "Post a 15-minute progress checkpoint (OpenClawd → Cursor)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "progress": {"type": "string"},
                        "partial_d": {"type": "number"},
                        "partial_p": {"type": "number"},
                        "gpu_memory_gb": {"type": "number"},
                        "anomalies": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["model", "progress", "partial_d", "partial_p", "gpu_memory_gb"]
                }
            },
            {
                "name": "get_checkpoints",
                "description": "Get recent checkpoints (Cursor reviews OpenClawd progress)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 10},
                        "since": {"type": "string"}
                    }
                }
            },
            {
                "name": "post_finding",
                "description": "Post a finding (result, insight, concern, suggestion)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["cursor", "openclawd"]},
                        "finding_type": {"type": "string", "enum": ["result", "insight", "concern", "suggestion"]},
                        "content": {"type": "string"},
                        "evidence": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
                    },
                    "required": ["source", "finding_type", "content"]
                }
            },
            {
                "name": "get_findings",
                "description": "Get findings from either agent",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "finding_type": {"type": "string"},
                        "unacknowledged_only": {"type": "boolean"}
                    }
                }
            },
            {
                "name": "suggest_experiment",
                "description": "Cursor suggests next experiment to OpenClawd",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "experiment": {"type": "string"},
                        "model": {"type": "string"},
                        "rationale": {"type": "string"},
                        "priority": {"type": "integer"},
                        "config_path": {"type": "string"}
                    },
                    "required": ["experiment", "model", "rationale", "priority"]
                }
            },
            {
                "name": "get_experiment_status",
                "description": "Get current experiment status",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "verify_logging",
                "description": "Verify experiment artifacts are properly logged",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "results_path": {"type": "string"}
                    },
                    "required": ["results_path"]
                }
            },
            {
                "name": "start_experiment",
                "description": "Mark experiment as started",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "experiment": {"type": "string"},
                        "model": {"type": "string"},
                        "config_path": {"type": "string"}
                    },
                    "required": ["experiment", "model", "config_path"]
                }
            },
            {
                "name": "end_experiment",
                "description": "Mark experiment as ended",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "results_path": {"type": "string"},
                        "success": {"type": "boolean"},
                        "summary": {"type": "object"}
                    },
                    "required": ["results_path", "success"]
                }
            }
        ]


def handle_request(server: MCPServer, request: Dict) -> Dict:
    """Handle MCP JSON-RPC request."""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "mi-monitor",
                    "version": "1.0.0"
                },
                "capabilities": {
                    "tools": {}
                }
            }
        }
    
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": server.get_tools()}
        }
    
    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        
        # Dispatch to tool
        tool_fn = getattr(server, tool_name, None)
        if tool_fn:
            try:
                result = tool_fn(**tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": str(e)}
                }
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
            }
    
    elif method == "notifications/initialized":
        return None  # No response needed
    
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"}
        }


def main():
    """Run MCP server on stdio."""
    server = MCPServer()
    
    # Read from stdin, write to stdout (MCP protocol)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            request = json.loads(line)
            response = handle_request(server, request)
            if response:
                print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            error = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"}
            }
            print(json.dumps(error), flush=True)


if __name__ == "__main__":
    main()
