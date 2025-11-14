#!/bin/bash
# Quick job status check script

echo "Job Status:"
echo "=========================================="
squeue -u sturgis -j 780

echo ""
echo "Latest output (last 30 lines):"
echo "=========================================="
tail -n 30 /data/spack/users/sturgis/winston_transfer/logs/soc_classify_780.out 2>/dev/null || echo "Log file not yet created"

echo ""
echo "Any errors?"
echo "=========================================="
tail -n 20 /data/spack/users/sturgis/winston_transfer/logs/soc_classify_780.err 2>/dev/null || echo "No errors yet"
