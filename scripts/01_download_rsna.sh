#!/usr/bin/env bash
# Phase 0 / step 2: download the RSNA Pneumonia Detection Challenge data into dataset/rsna.
# Prerequisites:
#   1) Accept the competition rules once on the Kaggle website:
#      https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules
#   2) Kaggle API token at ~/.kaggle/kaggle.json (chmod 600).
# Run from the repository root with env "xfi" active:   bash scripts/01_download_rsna.sh
set -euo pipefail

DEST=dataset/rsna
mkdir -p "$DEST"

if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ ! -f "$HOME/.kaggle/access_token" ]; then
  echo "[error] ~/.kaggle/kaggle.json not found"; exit 1
fi
chmod 600 "$HOME"/.kaggle/* 2>/dev/null || true

kaggle competitions download -c rsna-pneumonia-detection-challenge -p "$DEST"
python -m zipfile -e "$DEST/rsna-pneumonia-detection-challenge.zip" "$DEST"

echo "[check] labels rows (expected 30227 + header):"; wc -l "$DEST/stage_2_train_labels.csv"
echo "[check] train DICOM count (expected 26684):"; ls "$DEST/stage_2_train_images" | wc -l
sha256sum "$DEST/stage_2_train_labels.csv"
