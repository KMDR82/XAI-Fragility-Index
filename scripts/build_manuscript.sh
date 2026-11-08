#!/usr/bin/env bash
# Build the revised manuscript and the supplementary file:
# numbers/tables from the analysis outputs, figures from fig/manuscript, then LaTeX, then the checks.
#   bash scripts/build_manuscript.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/make_numbers_tex.py
mkdir -p manuscript/figures
cp fig/manuscript/*.pdf manuscript/figures/
cp manuscript/v1/wlpeerj.cls manuscript/
cd manuscript
if command -v pdflatex >/dev/null; then
  pdflatex -interaction=nonstopmode main >/dev/null || true
  bibtex main >/dev/null || true
  pdflatex -interaction=nonstopmode main >/dev/null || true
  pdflatex -interaction=nonstopmode main >/dev/null || true
  grep -E "^!|undefined" main.log | sort | uniq -c || echo "no LaTeX errors"
  echo "-> manuscript/main.pdf ($(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $2}') pages)"
  # the supplementary file refers to figure and table numbers of main.pdf (xr), so it is built afterwards
  pdflatex -interaction=nonstopmode supplementary >/dev/null || true
  pdflatex -interaction=nonstopmode supplementary >/dev/null || true
  grep -E "^!|undefined" supplementary.log | sort | uniq -c || echo "no LaTeX errors (supplementary)"
  echo "-> manuscript/supplementary.pdf ($(pdfinfo supplementary.pdf 2>/dev/null | awk '/Pages/{print $2}') pages)"
else
  echo "pdflatex not found: upload the manuscript/ folder to Overleaf, or: sudo apt install texlive-latex-extra texlive-fonts-recommended texlive-science"
fi
cd ..
python scripts/check_manuscript.py || true
