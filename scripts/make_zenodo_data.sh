#!/usr/bin/env bash
# Builds the data deposit for Zenodo (model weights, per-patient results, frozen splits, training logs,
# analysis tables) in zenodo_data/, with SHA256SUMS.txt. Run from the repository root after Phase 6b.
#   bash scripts/make_zenodo_data.sh
# Not included: the RSNA images and the stored example maps (output/phase3/*/maps/*.npz contain image
# crops); the RSNA data are obtained from Kaggle under the competition terms.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=zenodo_data
rm -rf "$OUT" zenodo_results_tables.zip; mkdir -p "$OUT"/{results,splits,training,tables}

declare -A W=( [A]=model/modelA_kaggle_v1.pth [Arep]=model/Arep_best.pth [Anorm]=model/Anorm_best.pth [B]=model/B_best.pth )
for m in A Arep Anorm B; do
  test -f "${W[$m]}" || { echo "missing ${W[$m]}"; exit 1; }
  cp "${W[$m]}" "$OUT/model_${m}.pth"
  want=$(python -c "import json;print(json.load(open('output/phase3/xfi_$m/manifest.json'))['weights_sha256'])")
  got=$(sha256sum "$OUT/model_${m}.pth" | cut -d' ' -f1)
  [ "$want" = "$got" ] || { echo "SHA-256 of model $m differs from the run manifest"; exit 1; }
  d="$OUT/results/xfi_$m"; mkdir -p "$d"
  gzip -n -c "output/phase3/xfi_$m/results_main.csv" > "$d/results_main.csv.gz"
  gzip -n -c "output/phase3/xfi_$m/results_base.csv" > "$d/results_base.csv.gz"
  cp "output/phase3/xfi_$m/manifest.json" "output/phase3/xfi_$m/run.log" "$d/"
done
cp output/phase0/split_train.csv output/phase0/split_test.csv output/phase2/split_val.csv output/phase2/split_train_fit.csv "$OUT/splits/"
for m in Arep Anorm B; do
  mkdir -p "$OUT/training/train_$m"
  for f in epochs.csv val_probs.csv test_probs.csv summary.json train.log; do
    [ -f "output/phase2/train_$m/$f" ] && cp "output/phase2/train_$m/$f" "$OUT/training/train_$m/"
  done
done
cp output/phase4/R*.csv "$OUT/tables/"
# the local backup folder is not part of the method; its value is blanked in the deposited copy
sed -E 's#^(\s*drive_backup_dir:).*#\1 "SET_ME"#' configs/config.yaml > "$OUT/config.yaml"
cp requirements-r1.txt "$OUT/"
cp docs/DATA_README.md "$OUT/README.md"
# checksums of the zip contents (inside the zip), then the zip, then checksums of the uploaded files
(cd "$OUT" && find results splits training tables config.yaml requirements-r1.txt -type f -print0 | sort -z | xargs -0 sha256sum > CONTENTS_SHA256.txt)
(cd "$OUT" && zip -qr zenodo_results_tables.zip CONTENTS_SHA256.txt config.yaml requirements-r1.txt results splits training tables)
(cd "$OUT" && rm -rf results splits training tables CONTENTS_SHA256.txt config.yaml requirements-r1.txt)
(cd "$OUT" && sha256sum model_A.pth model_Arep.pth model_Anorm.pth model_B.pth zenodo_results_tables.zip README.md > SHA256SUMS.txt)
echo "Upload these 7 files of $OUT/ to Zenodo:"
ls -la "$OUT" | awk 'NR>1 && $1 !~ /^d/ {printf "  %-32s %8.1f MB\n", $NF, $5/1048576}'
