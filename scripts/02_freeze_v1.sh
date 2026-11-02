#!/usr/bin/env bash
# Phase 0 / step 3: freeze the submitted state and open the revision branch (D-22).
# Run from the repository root (clean working tree, on main):   bash scripts/02_freeze_v1.sh
set -euo pipefail

git fetch origin
git checkout main
git pull --ff-only origin main

if git rev-parse -q --verify "refs/tags/v1.0-submitted" >/dev/null; then
  echo "[info] tag v1.0-submitted already exists"
else
  git tag -a v1.0-submitted -m "State of the repository at PeerJ submission #137915 (before R1 revision)"
  git push origin v1.0-submitted
fi

git checkout -B revision-r1

# Archive the v1 CSV as a legacy audit artifact (ART-001). It is git-ignored; its hash is logged.
mkdir -p output/v1_legacy
python3 -m zipfile -e logs/full_experiment_results_xfi.zip output/v1_legacy
echo "[ART-001] sha256 of v1 CSV:"
sha256sum output/v1_legacy/full_experiment_results_xfi.csv | tee output/v1_legacy/SHA256SUMS.txt
echo "[check] rows (expected 192132 + header):"; wc -l output/v1_legacy/full_experiment_results_xfi.csv
