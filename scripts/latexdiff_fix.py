"""Post-processing of the latexdiff output for the tracked-changes PDF.

    python scripts/latexdiff_fix.py <diffdir>

1. The PeerJ class stores the abstract before \\begin{document}, where latexdiff does not
   compare text. The two abstracts are diffed separately and the result is put back.
2. The latexdiff preamble is moved in front of the abstract, so that its commands exist there.
3. A numeric macro (\\xspace) directly followed by a \\DIF... command loses its trailing space;
   an explicit control space is inserted.
"""
import re
import subprocess
import sys
from pathlib import Path

d = Path(sys.argv[1])
ABS = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.S)
HDR = "\\documentclass{article}\n\\begin{document}\n"

for src, dst in [("old.tex", "abs_old.tex"), ("new.tex", "abs_new.tex")]:
    (d / dst).write_text(HDR + ABS.search((d / src).read_text(encoding="utf-8")).group(1) + "\n\\end{document}\n", encoding="utf-8")
res = subprocess.run(["latexdiff", "--type=UNDERLINE", "--math-markup=whole", "abs_old.tex", "abs_new.tex"],
                     cwd=d, capture_output=True, text=True, check=True)
abs_body = re.search(r"\\begin\{document\}(.*)\\end\{document\}", res.stdout, re.S).group(1)

t = (d / "diff.tex").read_text(encoding="utf-8")
m = ABS.search(t)
t = t[:m.start(1)] + abs_body + t[m.end(1):]
i = t.index("%DIF PREAMBLE EXTENSION ADDED BY LATEXDIFF")
j = t.index("%DIF END PREAMBLE EXTENSION ADDED BY LATEXDIFF")
pre = t[i:j]
t = t[:i] + t[j:]
k = t.index("\\begin{abstract}")
t = t[:k] + pre + "\n" + t[k:]

names = set(re.findall(r"\\newcommand\{\\(\w+)\}", (d / "numbers.tex").read_text(encoding="utf-8")))
t = re.sub(r"\\(\w+)[ \t]+(?=\\DIF)", lambda m: "\\" + m.group(1) + ("\\ " if m.group(1) in names else " "), t)
(d / "diff.tex").write_text(t, encoding="utf-8")
print("diff.tex post-processed")
