#!/bin/bash
# Monitor H31 validation progress on RunPod

echo "Monitoring H31 validation run..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    ssh -p 47660 -i ~/.ssh/id_ed25519 root@195.26.233.44 << 'EOF'
cd /workspace/mech-interp-latent-lab-phase1
if [ -f h31_validation_run.log ]; then
    echo "=== Last 10 lines of log ==="
    tail -10 h31_validation_run.log
    echo ""
    echo "=== Process status ==="
    ps aux | grep "h31_validation_n50.py" | grep -v grep || echo "Process not running"
    echo ""
    if [ -f results/h31_validation/h31_validation_n50.csv ]; then
        echo "=== CSV exists! Row count ==="
        wc -l results/h31_validation/h31_validation_n50.csv
    fi
else
    echo "Log file not found yet..."
fi
EOF
    echo "---"
    sleep 10
done









