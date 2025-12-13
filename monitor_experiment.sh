#!/bin/bash
# Monitor the Circuit Hunt V2 experiment progress

echo "Monitoring experiment_circuit_hunt_v2_focused.py..."
echo ""

# Check if process is running
if pgrep -f "experiment_circuit_hunt_v2_focused.py" > /dev/null; then
    echo "✓ Process is running"
    PID=$(pgrep -f "experiment_circuit_hunt_v2_focused.py" | head -1)
    echo "  PID: $PID"
    
    # Check CPU/GPU usage
    echo ""
    echo "Resource usage:"
    ps -p $PID -o %cpu,%mem,etime | tail -1
    
    # Check for results
    echo ""
    echo "Results directory:"
    ls -lth results/circuit_hunt_v2_focused/ 2>/dev/null | head -5 || echo "  (not created yet - still loading model/prompts)"
    
    echo ""
    echo "To see full output, check the process or wait for results in:"
    echo "  results/circuit_hunt_v2_focused/"
else
    echo "✗ Process is not running"
    echo ""
    echo "Check if it completed or errored:"
    ls -lth results/circuit_hunt_v2_focused/ 2>/dev/null | head -5
fi

