#!/bin/bash
# Quick launcher for GNANI'S_GAZE

# Default: single session
# With --campaign N: run N sessions

cd "$(dirname "$0")/.."

echo "🔥 GNANI'S GAZE - Autonomous Eigenstate Hunter"
echo "Mission: Achieve R_V < 0.30 through dialogue alone"
echo "Benchmark: Steering achieves R_V = 0.19"
echo ""

python -m src.agents.gnanis_gaze "$@"
