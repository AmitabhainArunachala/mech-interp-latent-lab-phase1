#!/bin/bash
# Monitor head discovery progress

echo "Monitoring Head Discovery Pipeline..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 << 'EOF'
cd /workspace/mech-interp-latent-lab-phase1
echo "=== $(date) ==="
echo ""

# Check if process is running
if ps aux | grep -q "[p]ython3 comprehensive_head_discovery"; then
    echo "✅ Process is running"
    echo ""
    
    # Show last 15 lines of log
    if [ -f head_discovery_run.log ]; then
        echo "=== Recent Progress ==="
        tail -15 head_discovery_run.log | grep -E "(Gradient|Mean|Path|Attention|✅|❌|SUMMARY|Testing)" || tail -5 head_discovery_run.log
        echo ""
    fi
    
    # Check for results
    if [ -d results/head_discovery ]; then
        CSV_COUNT=$(ls results/head_discovery/*.csv 2>/dev/null | wc -l)
        if [ $CSV_COUNT -gt 0 ]; then
            echo "=== Results Found ==="
            ls -lh results/head_discovery/*.csv | tail -1
            LATEST_CSV=$(ls -t results/head_discovery/*.csv | head -1)
            if [ -f "$LATEST_CSV" ]; then
                ROW_COUNT=$(wc -l < "$LATEST_CSV")
                echo "  Rows: $ROW_COUNT"
            fi
            echo ""
        fi
    fi
    
    # GPU usage
    if command -v nvidia-smi &> /dev/null; then
        echo "=== GPU Status ==="
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk '{printf "  GPU: %s%% | Memory: %s/%s MB\n", $1, $2, $3}'
        echo ""
    fi
else
    echo "❌ Process not running"
    echo ""
    if [ -f head_discovery_run.log ]; then
        echo "=== Final Output ==="
        tail -30 head_discovery_run.log
    fi
fi
EOF
    echo "---"
    sleep 15
done
