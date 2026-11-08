"""Paragraph-level tracked changes for a heavily rewritten manuscript.

    python scripts/latexdiff_blocks.py <diffdir> [threshold]

A word-level latexdiff of a largely rewritten text interleaves deleted and added fragments
until nothing is readable. This script compares the two versions paragraph by paragraph:

* identical paragraphs are copied unchanged;
* a revised paragraph that still shares at least `threshold` (default 0.5) of its words with
  its counterpart in the original is diffed word by word with latexdiff;
* every other paragraph is shown as a whole: the original paragraph struck through, the new
  paragraph underlined (both produced by latexdiff, so math, citations and environments stay valid).

Input in <diffdir>: old.tex, new.tex (with \\input files resolvable from <diffdir>) and the
diff.tex of a normal latexdiff run, whose preamble (including the latexdiff commands) is reused.
Output: <diffdir>/diff.tex (body replaced).
"""
import difflib
import re
import subprocess
import sys
from pathlib import Path

D = Path(sys.argv[1])
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
OPTS = ["--graphics-markup=0", "--math-markup=whole", "--type=UNDERLINE",
        "--config", r"PICTUREENV=(?:picture|DIFnomarkup|algorithmic|algorithm|tabular|figure|table)[\w\d*@]*"]


def flatten(text, base, depth=0):
    def rep(m):
        p = base / m.group(1)
        p = p if p.suffix == ".tex" else p.with_suffix(".tex")
        return flatten(p.read_text(encoding="utf-8"), base, depth + 1) if p.exists() and depth < 5 else m.group(0)
    return re.sub(r"(?<!%)\\input\{([^}]+)\}", rep, text)


def body(text):
    i, j = text.index("\\begin{document}") + len("\\begin{document}"), text.index("\\end{document}")
    return text[i:j]


def strip_comments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", l) if not l.lstrip().startswith("%") else "" for l in s.split("\n"))


def blocks(text):
    """Paragraphs separated by blank lines, merged so that no environment is split."""
    out, cur, depth = [], [], 0
    for chunk in re.split(r"\n\s*\n", strip_comments(text)):
        if not chunk.strip():
            continue
        cur.append(chunk)
        depth += len(re.findall(r"\\begin\{", chunk)) - len(re.findall(r"\\end\{", chunk))
        if depth <= 0:
            out.append("\n".join(cur).strip()); cur, depth = [], 0
    if cur:
        out.append("\n".join(cur).strip())
    return out


def words(s):
    return re.sub(r"\\[A-Za-z]+\*?|[{}$\\~]", " ", s).split()


def sim(a, b):
    return difflib.SequenceMatcher(None, words(a), words(b), autojunk=False).ratio()


HDR = "\\documentclass{article}\n\\begin{document}\n"
_n = [0]


def ldiff(a, b):
    _n[0] += 1
    fa, fb = D / f"_p{_n[0]}a.tex", D / f"_p{_n[0]}b.tex"
    fa.write_text(HDR + a + "\n\\end{document}\n", encoding="utf-8")
    fb.write_text(HDR + b + "\n\\end{document}\n", encoding="utf-8")
    r = subprocess.run(["latexdiff", *OPTS, fa.name, fb.name], cwd=D, capture_output=True, text=True)
    fa.unlink(); fb.unlink()
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return body(r.stdout).strip()


old = blocks(body(flatten((D / "old.tex").read_text(encoding="utf-8"), D)))
new = blocks(body(flatten((D / "new.tex").read_text(encoding="utf-8"), D)))

out, stats = [], dict(same=0, word=0, block_old=0, block_new=0)
sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op == "equal":
        out += new[j1:j2]; stats["same"] += j2 - j1; continue
    O, N = old[i1:i2], new[j1:j2]
    # monotone pairing of sufficiently similar paragraphs inside the changed region
    pairs, start = [], 0
    for jn, b in enumerate(N):
        best, bi = 0.0, None
        for io in range(start, len(O)):
            s = sim(O[io], b)
            if s > best:
                best, bi = s, io
        if bi is not None and best >= THR:
            pairs.append((bi, jn)); start = bi + 1
    # region in reading order: unmatched originals (deleted), unmatched new (added), and word-level pairs
    region = []
    io = 0; jo = 0
    for bi, jn in pairs + [(len(O), len(N))]:
        for k in range(io, bi):
            region.append(("del", O[k]))
        for k in range(jo, jn):
            region.append(("add", N[k]))
        if bi < len(O) and jn < len(N):
            region.append(("pair", O[bi], N[jn]))
        io, jo = bi + 1, jn + 1
    for x in region:
        if x[0] == "del":
            out.append(ldiff(x[1], "")); stats["block_old"] += 1
        elif x[0] == "add":
            out.append(ldiff("", x[1])); stats["block_new"] += 1
        else:
            out.append(ldiff(x[1], x[2])); stats["word"] += 1

t = (D / "diff.tex").read_text(encoding="utf-8")
i, j = t.index("\\begin{document}") + len("\\begin{document}"), t.index("\\end{document}")
(D / "diff.tex").write_text(t[:i] + "\n" + "\n\n".join(out) + "\n" + t[j:], encoding="utf-8")
print(f"paragraphs: {stats['same']} unchanged, {stats['word']} word-level, {stats['block_old']} deleted and {stats['block_new']} added as a whole (threshold {THR})")
