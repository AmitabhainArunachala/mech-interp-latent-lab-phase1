#!/bin/bash
# Quick check of experiment progress

PID=$(pgrep -f "experiment_circuit_hunt_v2_focused.py" | head -1)

if [ -z "$PID" ]; then
    echo "✗ Process not running"
    echo ""
    echo "Checking for results..."
    ls -lth results/circuit_hunt_v2_focused/*.json 2>/dev/null | head -3
    exit 0
fi

echo "✓ Process running (PID: $PID)"
echo ""

# Get runtime
RUNTIME=$(ps -p $PID -o etime= | tr -d ' ')
echo "Runtime: $RUNTIME"

# Check CPU/Memory
ps -p $PID -o %cpu,%mem= | tail -1 | awk '{printf "CPU: %.1f%%, Memory: %.1f%%\n", $1, $2}'

echo ""
echo "Results directory:"
if [ -d "results/circuit_hunt_v2_focused" ]; then
    FILE_COUNT=$(ls -1 results/circuit_hunt_v2_focused/*.json 2>/dev/null | wc -l | tr -d ' ')
    if [ "$FILE_COUNT" -gt 0 ]; then
        echo "  ✓ Found $FILE_COUNT result file(s):"
        ls -lth results/circuit_hunt_v2_focused/*.json | head -3 | awk '{print "    " $9 " (" $5 ")"}'
    else
        echo "  ⏳ No results yet (still initializing or running experiments)"
    fi
else
    echo "  ⏳ Directory not created yet"
fi

echo ""
echo "Note: Running on CPU (CUDA=False) will be VERY slow."
echo "      Model loading: ~2-5 min, Experiments: hours"

