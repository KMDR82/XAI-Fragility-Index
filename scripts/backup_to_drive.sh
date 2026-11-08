#!/usr/bin/env bash
# Mirror everything produced in the revision (except the raw dataset) to Google Drive (D-24).
# Uses drive_backup_dir from configs/config.yaml. Run after every completed step:
#   bash scripts/backup_to_drive.sh
set -euo pipefail

DEST=$(python -c "import yaml;print(yaml.safe_load(open('configs/config.yaml'))['paths']['drive_backup_dir'])")
if [ "$DEST" = "SET_ME" ] || [ -z "$DEST" ]; then
  echo "[error] set paths.drive_backup_dir in configs/config.yaml"; exit 1
fi
PARENT=$(dirname "$DEST")
if [ ! -d "$PARENT" ]; then
  echo "[error] $PARENT not found: is Google Drive mounted at /mnt/g?"; exit 1
fi
mkdir -p "$DEST"

rsync -r --checksum --info=stats1 \
  --exclude 'dataset/' --exclude '.git/' --exclude '__pycache__/' --exclude '.ipynb_checkpoints/' \
  ./ "$DEST/"

echo "$(date -Iseconds) backup -> $DEST" >> output/backup_history.log
echo "[done] backup to $DEST"
