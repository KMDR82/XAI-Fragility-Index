#!/usr/bin/env bash
# Night run 1 (Phase 2/3): image cache + training of Arep, Anorm, B. Unattended; each step continues even if a previous one fails.
# Start from the repo root:
#   nohup bash scripts/run_night1_training.sh > output/phase2/night1.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
eval "$(conda shell.bash hook)"; conda activate xfi
export PYTHONPATH="$PWD/src"
mkdir -p output/phase2
echo "=== night1 start $(date -Iseconds) | git $(git rev-parse --short HEAD) ==="
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv

for M in Arep Anorm B; do
  echo "=== $M start $(date -Iseconds) ==="
  python -m xfi.train --model "$M" --workers 12 && echo "=== $M OK $(date -Iseconds) ===" || echo "=== $M FAILED $(date -Iseconds) ==="
done

sha256sum model/*_best.pth model/*_epoch*.pth 2>/dev/null > output/phase2/weights_sha256.txt
bash scripts/backup_to_drive.sh || echo "backup failed"
echo "=== night1 end $(date -Iseconds) ==="
touch output/phase2/NIGHT1_DONE
