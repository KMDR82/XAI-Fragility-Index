"""Consistency checks on the built manuscript.

Run after scripts/build_manuscript.sh:
    python scripts/check_manuscript.py
Exit code 0 = all checks passed, 1 = at least one FAIL.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))


def tex_sources():
    files = [MS / "main.tex"] + sorted((MS / "sections").glob("*.tex")) + sorted((MS / "tables").glob("*.tex"))
    return {f: f.read_text(encoding="utf-8") for f in files if f.exists()}


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


src = tex_sources()
body = "\n".join(strip_comments(s) for s in src.values())
log = (MS / "main.log").read_text(encoding="latin-1") if (MS / "main.log").exists() else ""
blg = (MS / "main.blg").read_text(encoding="latin-1") if (MS / "main.blg").exists() else ""
pdf = MS / "main.pdf"

# 1. build artefacts
check("main.pdf exists", pdf.exists())
errs = [l for l in log.splitlines() if l.startswith("!")]
check("no LaTeX errors", not errs, "; ".join(errs[:3]))
over = re.findall(r"Overfull \\hbox \(([\d.]+)pt", log)
big = [o for o in over if float(o) > 1.0]
check("no overfull hbox > 1pt", not big, f"{len(big)} found: {big[:5]}")
undef_ref = sorted(set(re.findall(r"Reference `([^']+)' on page", log)))
check("no undefined references", not undef_ref, ", ".join(undef_ref))
undef_cit = sorted(set(re.findall(r"Citation `([^']+)' on page", log)))
check("no undefined citations", not undef_cit, ", ".join(undef_cit))
check("no 'multiply defined' labels", "multiply defined" not in log)
bib_warn = [l for l in blg.splitlines() if l.startswith("Warning")]
check("no BibTeX warnings", not bib_warn, "; ".join(bib_warn[:3]))

# 2. citations vs bibliography
cited = set()
for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", body):
    cited.update(k.strip() for k in m.group(1).split(","))
bib = (MS / "references.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
check("every cited key is in references.bib", cited <= keys, ", ".join(sorted(cited - keys)))
unused = sorted(keys - cited)
check("bib entries not cited (info only)", True, f"{len(unused)}: {', '.join(unused)}")
noyear = [k for k in cited if not re.search(r"@\w+\{" + re.escape(k) + r",[^@]*year\s*=", bib)]
check("every cited entry has a year", not noyear, ", ".join(noyear))

# 3. labels and refs
labels = set(re.findall(r"\\label\{([^}]+)\}", body))
refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", body))
check("every \\ref has a \\label", refs <= labels, ", ".join(sorted(refs - labels)))
check("labels never referenced (info only)", True, ", ".join(sorted(labels - refs)))

# 4. figures exist
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
missing = [f for f in figs if not any((MS / (f + ext)).exists() for ext in ("", ".pdf", ".png"))]
check("every included figure file exists", not missing, ", ".join(missing))

# 5. macros from numbers.tex are all defined
defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", (MS / "numbers.tex").read_text(encoding="utf-8")))
used = set(re.findall(r"\\([A-Za-z]+)", body))
num_like = {u for u in used if re.match(r"^(n|auc|ap|ece|brier|acc|prec|rec|spec|fone|bacc|d|xfi|cross|rand|p|r|share|slope|flip|pos|iou|drop|bg|eout|area|rho|base|minor|sign|drift|thr|best|epochs|temp|retrain|pct)[A-Z]", u)}
latex_cmds = {"pdfoutput", "newcommand", "renewcommand", "providecommand", "includegraphics", "paragraph", "par"}
undef_macros = sorted(u for u in num_like - defined - latex_cmds if not re.match(r"^(par|ref|eqref|label)", u))
check("all numeric macros defined in numbers.tex", not undef_macros, ", ".join(undef_macros))

# 6. text-level checks on the PDF
if pdf.exists():
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
    leftovers = sorted(set(re.findall(r"\\[A-Za-z]{3,}", txt)))
    check("no raw \\macro names in PDF text", not leftovers, ", ".join(leftovers[:10]))
    check("no '??' (broken refs) in PDF", "??" not in txt)
    pages = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", pages)
    check("page count (info only)", True, m.group(1) if m else "?")
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src[MS / "main.tex"], re.S)
    if m:
        words = len(re.sub(r"\\[A-Za-z]+|[{}$]", " ", m.group(1)).split())
        check("abstract word count (info only)", True, str(words))

# 7. style checks on the source
def _no_verbatim(t):
    return re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", lambda m: "\n" * m.group(0).count("\n"), t, flags=re.S)
emd = [(f.name, i + 1) for f, s in src.items() for i, l in enumerate(_no_verbatim(s).splitlines()) if "---" in strip_comments(l) or "\u2014" in l]
check("no em-dashes in source", not emd, str(emd[:5]))
wm = [(f.name, i + 1) for f, s in src.items() for i, l in enumerate(s.splitlines()) if re.search(r"claude|anthropic|chatgpt|generated (by|with) ai", l, re.I)]
check("no assistant/AI watermark text", not wm, str(wm[:5]))
todo = [(f.name, i + 1) for f, s in src.items() for i, l in enumerate(s.splitlines()) if re.search(r"\bTODO\b|\bXXX\b|\bTBD\b|PLACEHOLDER", strip_comments(l))]
check("no TODO/TBD/placeholder markers", not todo, str(todo[:5]))
orcid = "0000-0003-4836-2310"
check("author ORCID present", orcid in src[MS / "main.tex"])
check("affiliation updated", "Gönen Vocational School" in src[MS / "main.tex"] or "G\\\"{o}nen" in src[MS / "main.tex"])

# 8. supplementary file
sup_tex = MS / "supplementary.tex"
if sup_tex.exists():
    s_src = strip_comments(sup_tex.read_text(encoding="utf-8"))
    s_log = (MS / "supplementary.log").read_text(encoding="latin-1") if (MS / "supplementary.log").exists() else ""
    s_pdf = MS / "supplementary.pdf"
    check("supplementary.pdf exists", s_pdf.exists())
    s_err = [l for l in s_log.splitlines() if l.startswith("!")]
    check("supplementary: no LaTeX errors", not s_err, "; ".join(s_err[:3]))
    s_over = [o for o in re.findall(r"Overfull \\[hv]box \(([\d.]+)pt", s_log) if float(o) > 1.0]
    check("supplementary: no overfull box > 1pt", not s_over, str(s_over[:5]))
    s_undef = sorted(set(re.findall(r"Reference `([^']+)' on page", s_log)))
    check("supplementary: all refs resolved (incl. main-*)", not s_undef, ", ".join(s_undef))
    s_figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s_src)
    s_miss = [f for f in s_figs if not any((MS / (f + ext)).exists() for ext in ("", ".pdf", ".png"))]
    check("supplementary: every figure file exists", not s_miss, ", ".join(s_miss))
    s_tabs = re.findall(r"\\input\{(tables/[^}]+)\}", s_src)
    t_miss = [t for t in s_tabs if not (MS / (t + ".tex")).exists()]
    check("supplementary: every table file exists", not t_miss, ", ".join(t_miss))
    n_tab, n_fig = s_src.count("\\begin{table}"), s_src.count("\\begin{figure}")
    want_t = {int(x) for x in re.findall(r"Table~S(\d+)", body)}
    want_f = {int(x) for x in re.findall(r"Figs?\.~S(\d+)", body)}
    check("main text refers only to existing Tables S#", max(want_t, default=0) <= n_tab, f"main refers to S{sorted(want_t)}, supplement has {n_tab}")
    check("main text refers only to existing Figs S#", max(want_f, default=0) <= n_fig, f"main refers to S{sorted(want_f)}, supplement has {n_fig}")
    if s_pdf.exists():
        s_txt = subprocess.run(["pdftotext", str(s_pdf), "-"], capture_output=True, text=True).stdout
        s_left = sorted(set(re.findall(r"\\[A-Za-z]{3,}", s_txt)))
        check("supplementary: no raw \\macro names / '??'", not s_left and "??" not in s_txt, ", ".join(s_left[:10]))
    check("supplementary: no assistant/AI watermark text", not re.search(r"claude|anthropic|chatgpt", s_src, re.I))

# 9. fonts: Type 3 fonts are often rejected by journal production systems
for p in [pdf, MS / "supplementary.pdf"]:
    if p.exists():
        f = subprocess.run(["pdffonts", str(p)], capture_output=True, text=True).stdout
        t3 = [l.split()[0] for l in f.splitlines()[2:] if "Type 3" in l]
        check(f"{p.name}: no Type 3 fonts", not t3, ", ".join(t3[:5]))

# report
w = max(len(n) for n, _, _ in results)
fails = 0
for name, ok, detail in results:
    tag = "PASS" if ok else "FAIL"
    fails += not ok
    print(f"[{tag}] {name.ljust(w)}  {detail}")
print(f"\n{len(results) - fails}/{len(results)} passed")
sys.exit(1 if fails else 0)
