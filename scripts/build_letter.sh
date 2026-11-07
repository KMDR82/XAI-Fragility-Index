#!/usr/bin/env bash
# Response letter: reads the line numbers from the compiled manuscript, then builds response/response_letter.pdf.
# Run scripts/build_manuscript.sh first (it produces manuscript/main.pdf and manuscript/numbers.tex).
set -euo pipefail
cd "$(dirname "$0")/.."
test -f manuscript/main.pdf || { echo "manuscript/main.pdf missing: run scripts/build_manuscript.sh first"; exit 1; }
python scripts/letter_lines.py manuscript/main.pdf response/letter_anchors.json response/letter_lines.tex
cd response
for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error response_letter.tex > /dev/null || { grep -A3 "^!" response_letter.log | head -20; exit 1; }; done
grep -q "Undefined control sequence\|undefined" response_letter.log && { echo "WARNING: undefined macros/references:"; grep -B1 -A2 "Undefined control\|undefined" response_letter.log | head -20; }
n=$(grep -c '\\todo{' response_letter.tex || true)
[ "$n" -gt 0 ] && echo "NOTE: $n open \\todo item(s) in the letter (shown in red)"
echo "-> response/response_letter.pdf ($(pdfinfo response_letter.pdf | awk '/Pages/{print $2}') pages)"
