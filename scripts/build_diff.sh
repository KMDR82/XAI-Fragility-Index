#!/usr/bin/env bash
# Tracked-changes PDF: original submission (manuscript/v1/main.tex) against the revised manuscript.
# Run scripts/build_manuscript.sh first (it produces numbers.tex, tables/ and figures/).
#   bash scripts/build_diff.sh            -> word-level latexdiff (default)
#   MODE=block bash scripts/build_diff.sh -> paragraph-level: rewritten paragraphs shown whole
# Output: manuscript/diff/main_tracked_changes.pdf
set -euo pipefail
cd "$(dirname "$0")/.."
M=manuscript; D=$M/diff
command -v latexdiff >/dev/null || { echo "latexdiff not found: sudo apt install latexdiff"; exit 1; }
[ -f $M/numbers.tex ] || { echo "run scripts/build_manuscript.sh first"; exit 1; }
rm -rf $D && mkdir -p $D
cp -r $M/sections $M/tables $M/figures $M/numbers.tex $M/wlpeerj.cls $D/
cp $M/v1/main.tex $D/old.tex
cp $M/main.tex $D/new.tex
# one bibliography that holds the entries of both versions (deleted citations stay resolvable)
python - "$M/references.bib" "$M/v1/sample.bib" "$D/diffrefs.bib" <<'EOF'
import re, sys
new, old = open(sys.argv[1], encoding="utf-8").read(), open(sys.argv[2], encoding="utf-8").read()
keys = set(re.findall(r"@\w+\{([^,\s]+),", new))
extra = [e for e, k in re.findall(r"(@\w+\{([^,\s]+),.*?\n\}\s*\n)", old + "\n", re.S) if k not in keys]
open(sys.argv[3], "w", encoding="utf-8").write(new + "\n" + "".join(extra))
print(f"diffrefs.bib: {len(keys)} current + {len(extra)} entries cited only in the original")
EOF
sed -i 's/\\bibliography{[^}]*}/\\bibliography{diffrefs}/' $D/old.tex $D/new.tex
cd $D
# figures, tables and the algorithm are compared as whole blocks: latexdiff cannot mark changes inside them reliably
latexdiff --flatten --graphics-markup=0 --math-markup=whole --type=UNDERLINE \
  --config "PICTUREENV=(?:picture|DIFnomarkup|algorithmic|algorithm|tabular|figure|table)[\w\d*@]*" \
  old.tex new.tex > diff.tex 2> latexdiff.log
cd ../..
if [ "${MODE:-word}" = block ]; then python scripts/latexdiff_blocks.py $D 0.5; fi
python scripts/latexdiff_fix.py $D
cd $D
pdflatex -interaction=nonstopmode diff >/dev/null || true
bibtex diff >/dev/null || true
pdflatex -interaction=nonstopmode diff >/dev/null || true
pdflatex -interaction=nonstopmode diff >/dev/null || true
grep -E "^!" diff.log | sort | uniq -c || echo "no LaTeX errors"
mv diff.pdf main_tracked_changes.pdf
echo "-> $D/main_tracked_changes.pdf ($(pdfinfo main_tracked_changes.pdf | awk '/Pages/{print $2}') pages)"
