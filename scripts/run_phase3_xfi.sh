#!/usr/bin/env bash
# Phase 3: full canonical v2 XFI runs for Model A and Model B (5,337 test patients each).
# Start ONLY after notebooks/02_pilot.ipynb reports "GATE (P1, P2, P3): PASS".
#   nohup bash scripts/run_phase3_xfi.sh > output/phase3/phase3.log 2>&1 &
# Resumable: if interrupted, start it again; finished shards are skipped and reproduced bit-identically.
set -u
cd "$(dirname "$0")/.."
eval "$(conda shell.bash hook)"; conda activate xfi
export PYTHONPATH="$PWD/src"
mkdir -p output/phase3
echo "=== phase3 start $(date -Iseconds) | git $(git rev-parse --short HEAD) ==="
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv

for M in A B; do
  echo "=== XFI $M start $(date -Iseconds) ==="
  python -m xfi.run_xfi --model "$M" --workers 16 && echo "=== XFI $M OK $(date -Iseconds) ===" || echo "=== XFI $M FAILED $(date -Iseconds) ==="
  bash scripts/backup_to_drive.sh || echo "backup failed"
done
echo "=== phase3 end $(date -Iseconds) ==="
touch output/phase3/PHASE3_DONE
