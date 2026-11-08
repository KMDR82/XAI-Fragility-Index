"""Generate manuscript/numbers.tex: every number quoted in the revised manuscript, as a LaTeX macro.

    python scripts/make_numbers_tex.py            # writes manuscript/numbers.tex and manuscript/numbers.csv

Sources: output/phase4/*.csv (analysis tables), output/phase2/train_*/summary.json (classifiers),
output/phase2/split_val.csv and the RSNA label file (data set sizes), output/phase3/xfi_*/results_main.csv (map counts).
A macro whose source is missing is written as a visible red placeholder, so it cannot slip into the text unnoticed.
numbers.csv lists every macro with its value and source; Phase 9 checks the PDF against it.
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
T4, P2, P3 = ROOT / "output" / "phase4", ROOT / "output" / "phase2", ROOT / "output" / "phase3"
OUT = ROOT / "manuscript"
MODELS = {"A": "A", "Arep": "Arep", "Anorm": "Anorm", "B": "B"}
MLAB = {"A": "A", "Arep": "Arep", "Anorm": "Anorm", "B": "B"}  # macro-safe (letters only)
METH = {"GradCAM": "Grad", "EigenCAM": "Eigen"}
METHODS = list(METH)
TYPE = {"Gaussian": "Gauss", "Blur": "Blur", "SaltPepper": "SP"}
LEV = {1: "One", 2: "Two", 3: "Three"}
ROWS = []


def put(name, value, source):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    ROWS.append((name, value, source))


def neg(t):
    return t.replace("-", "$-$")


def num(x, d=3):
    return neg(f"{x:.{d}f}")


def big(n):
    return f"{int(n):,}".replace(",", "{,}")


def ci(est, lo, hi, d=3):
    return neg(f"{est:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]")


def pval(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 1e-4:
        e = int(np.floor(np.log10(p))) if p > 0 else -300
        return r"$<10^{%d}$" % (e + 1) if p > 0 else r"$<10^{-300}$"
    return f"{p:.4f}"


def missing(name, why):
    put(name, r"\textcolor{red}{\textbf{??}}", f"MISSING: {why}")


# ------------------------------------------------------------------ data set (Methods)
def sec_data():
    s = {}
    for M in ["Arep", "Anorm", "B"]:
        f = P2 / f"train_{M}" / "summary.json"
        if f.exists():
            s[M] = json.load(open(f))
    if not s:
        return missing("nTest", "no phase2 summary")
    n = next(iter(s.values()))["n"]
    put("nTrainFull", big(n["train_full"]), "summary.json n.train_full")
    put("nTrainFit", big(n["train_fit"]), "summary.json n.train_fit")
    put("nVal", big(n["val"]), "summary.json n.val")
    put("nTest", big(n["test"]), "summary.json n.test")
    put("nPatients", big(n["train_full"] + n["test"]), "train_full + test")
    val = pd.read_csv(P2 / "split_val.csv")
    put("nValPos", big(val.Target.sum()), "split_val.csv")
    put("pctValPos", num(100 * val.Target.mean(), 1), "split_val.csv")
    res = P3 / "xfi_A" / "results_main.csv"
    if res.exists():
        y = pd.read_csv(res, usecols=["Patient", "Label"]).drop_duplicates("Patient").Label
        put("nTestPos", big(y.sum()), "results_main A")
        put("nTestNeg", big(len(y) - y.sum()), "results_main A")
        put("pctTestPos", num(100 * y.mean(), 1), "results_main A")
        put("nMapsPerModel", big(len(y) * 10 * 2), "patients x (1 clean + 9 perturbed) x 2 methods")
        put("nMapsAll", big(len(y) * 10 * 2 * 4), "x 4 classifiers")
    else:
        missing("nTestPos", "results_main A")
    lab = ROOT / "dataset" / "rsna" / "stage_2_train_labels.csv"
    if lab.exists():
        L = pd.read_csv(lab).drop_duplicates("patientId")
        put("nPatientsPos", big(L.Target.sum()), "RSNA labels")
        put("pctPatientsPos", num(100 * L.Target.mean(), 1), "RSNA labels")
    else:
        missing("nPatientsPos", "RSNA label file")
    for M, d in s.items():
        m = MLAB[M]
        put(f"bestEpoch{m}", str(d["best_epoch"]), f"{M} summary.json")
        put(f"thr{m}", num(d["val_youden_threshold"]), f"{M} validation Youden threshold")
    if "B" in s:
        put("posWeightB", num(s["B"]["pos_weight"], 2), "B summary.json")
        put("epochsRunB", str(len(pd.read_csv(P2 / "train_B" / "epochs.csv"))), "B epochs.csv")


# ------------------------------------------------------------------ helpers for the results sections
def csv(name):
    f = T4 / f"{name}.csv"
    return pd.read_csv(f) if f.exists() else None


def results(M, cols):
    f = P3 / f"xfi_{M}" / "results_main.csv"
    return pd.read_csv(f, usecols=cols) if f.exists() else None


ORDER = ["A", "Arep", "Anorm", "B"]
TNAME = {"Gaussian": "Gaussian noise", "Blur": "Gaussian blur", "SaltPepper": "Salt-and-pepper"}
MNAME = {"A": "A", "Arep": "A-rep", "Anorm": "A-norm", "B": "B"}
TABLES = {}


def rng(vals, d=2):
    v = sorted(vals)
    return num(v[0], d), num(v[-1], d)


# ------------------------------------------------------------------ classifiers (Results 3.1, Table 2)
def sec_classifiers():
    r = csv("R1_classifiers")
    if r is None:
        return missing("aucA", "R1")
    tf = {"AUC": "auc", "average_precision": "ap", "ECE_10bins": "ece", "Brier": "brier"}
    op = {"accuracy": "acc", "precision": "prec", "recall": "rec", "specificity": "spec", "f1": "fone", "balanced_accuracy": "bacc"}
    rows = []
    for M in ORDER:
        q = r[r.model == M]
        if q.empty:
            continue
        for met, key in tf.items():
            x = q[q.metric == met].iloc[0]
            put(f"{key}{M}", num(x.estimate), f"R1 {M} {met}"); put(f"{key}{M}CI", ci(x.estimate, x.ci_low, x.ci_high), f"R1 {M} {met}")
        oper = q[q.threshold == "operating"]
        if oper.empty:
            oper = q[q.threshold == "0.5"]
        for met, key in op.items():
            x = oper[oper.metric == met].iloc[0]
            put(f"{key}{M}", num(x.estimate), f"R1 {M} {met} operating"); put(f"{key}{M}CI", ci(x.estimate, x.ci_low, x.ci_high), f"R1 {M} {met} operating")
            x5 = q[(q.threshold == "0.5") & (q.metric == met)].iloc[0]
            put(f"{key}{M}half", num(x5.estimate), f"R1 {M} {met} at 0.5")
    # Table 2
    lab = [("AUC", "auc"), ("Average precision", "ap"), ("ECE", "ece"), ("Brier score", "brier"), (None, None),
           ("Accuracy", "acc"), ("Precision", "prec"), ("Recall (sensitivity)", "rec"), ("Specificity", "spec"), ("F1 score", "fone"), ("Balanced accuracy", "bacc")]
    thr = {"A": "0.500"}
    for M in ORDER[1:]:
        f = P2 / f"train_{M}" / "summary.json"
        if f.exists():
            thr[M] = num(json.load(open(f))["val_youden_threshold"])
    head = " & ".join(f"Model {MNAME[M]}" for M in ORDER)
    body = []
    for name, key in lab:
        if name is None:
            body.append(r"\midrule")
            body.append("Threshold & " + " & ".join(thr.get(M, "--") for M in ORDER) + r" \\")
            continue
        body.append(name + " & " + " & ".join(f"\\{key}{M}CI" for M in ORDER) + r" \\")
    TABLES["T2_classifiers"] = ("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nMetric & " + head + " \\\\\n\\midrule\n" + "\n".join(body)
                                + "\n\\bottomrule\n\\end{tabular}\n")
    body = [name + " & " + " & ".join(f"\\{key}{M}half" for M in ORDER) + r" \\" for name, key in lab[5:]]
    TABLES["TS1_classifiers_threshold05"] = ("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nMetric & " + head + " \\\\\n\\midrule\n" + "\n".join(body)
                                             + "\n\\bottomrule\n\\end{tabular}\n")
    d = csv("R1b_paired_classifier_differences")
    if d is not None:
        nm = {"Anorm - Arep": "NormVsRep", "Arep - A": "RepVsA", "B - A": "BVsA", "B - Anorm": "BVsNorm"}
        for _, x in d.iterrows():
            k = nm.get(x.comparison)
            if k:
                put(f"d{x.metric.capitalize()}{k}", ci(x["diff"], x.ci_low, x.ci_high), f"R1b {x.comparison} {x.metric}")
    c = csv("R1c_temperature_scaling")
    if c is not None:
        for _, x in c.iterrows():
            put(f"eceTemp{x.model}", num(x.ece_test_temperature), "R1c"); put(f"ecePlatt{x.model}", num(x.ece_test_platt), "R1c")
            put(f"temp{x.model}", num(x.temperature, 2), "R1c")


# ------------------------------------------------------------------ XFI by condition, Grad vs Eigen (Results 3.2-3.4)
def sec_xfi():
    r3 = csv("R3_xfi_by_condition"); r2 = csv("R2_grad_vs_eigen"); r11 = csv("R11_baselines"); r4 = csv("R4_slopes")
    if r3 is None or r2 is None:
        return missing("xfiAGradAll", "R3/R2")
    models = [M for M in ORDER if M in set(r3.model)]
    for M in models:
        for m, mk in METH.items():
            s = r3[(r3.model == M) & (r3.method == m) & (r3.type != "pooled")]
            put(f"xfi{M}{mk}All", num(s.mean_xfi.mean()), "R3 mean of the 9 condition means (= patient-level mean)")
            for t, tk in TYPE.items():
                for l, lk in LEV.items():
                    x = s[(s.type == t) & (s.level == l)].iloc[0]
                    put(f"xfi{M}{mk}{tk}{lk}", num(x.mean_xfi), f"R3 {M} {m} {t} L{l}")
            for l, lk in LEV.items():
                x = r3[(r3.model == M) & (r3.method == m) & (r3.type == "pooled") & (r3.level == l)].iloc[0]
                put(f"xfi{M}{mk}Pooled{lk}", num(x.mean_xfi), f"R3 {M} {m} pooled L{l}")
            if r11 is not None:
                x = r11[(r11.model == M) & (r11.baseline == f"CrossPatient_{m}")].iloc[0]
                put(f"cross{M}{mk}", num(x.mean_xfi), "R11 cross-patient")
                x = r11[(r11.model == M) & (r11.baseline == f"ModelRandom_{m}")].iloc[0]
                put(f"rand{M}{mk}", num(x.mean_xfi), "R11 randomised model")
        a = r2[(r2.model == M) & (r2.type == "all")].iloc[0]
        put(f"dMean{M}", ci(a.mean_diff, a.mean_ci_low, a.mean_ci_high), "R2 all, mean Grad-Eigen")
        put(f"dMedian{M}", ci(a.median_diff, a.median_ci_low, a.median_ci_high), "R2 all, median Grad-Eigen")
        put(f"pAll{M}", pval(a.p), "R2 all Wilcoxon p"); put(f"rAll{M}", num(a.r, 2), "R2 all effect size")
        put(f"shareGradLower{M}", num(100 * a.share_grad_lower, 1), "R2 all share of patients with lower Grad-CAM XFI")
        c = r2[(r2.model == M) & (r2.type != "all")].copy(); c["level"] = c.level.astype(int)
        for _, x in c.iterrows():
            put(f"dMed{M}{TYPE[x.type]}{LEV[x.level]}", num(x.median_diff), "R2 condition median")
            put(f"pHolm{M}{TYPE[x.type]}{LEV[x.level]}", pval(x.p_holm), "R2 condition Holm p")
        sig = c[c.p_holm < 0.05]
        put(f"nGradLower{M}", str(int((sig.median_diff < 0).sum())), "R2 conditions with Holm p<0.05 and median<0")
        put(f"nEigenLower{M}", str(int((sig.median_diff > 0).sum())), "R2 conditions with Holm p<0.05 and median>0")
        put(f"nNS{M}", str(int((c.p_holm >= 0.05).sum())), "R2 conditions with Holm p>=0.05")
        if r4 is not None:
            x = r4[(r4.model == M) & (r4.type == "pooled")].iloc[0]
            put(f"slopeGrad{M}", num(x.GradCAM_slope), "R4 pooled"); put(f"slopeEigen{M}", num(x.EigenCAM_slope), "R4 pooled")
    # sign consistency of condition-wise medians across the four models
    c = r2[r2.type != "all"].copy(); c["level"] = c.level.astype(int)
    for t, tk in TYPE.items():
        for l, lk in LEV.items():
            v = c[(c.type == t) & (c.level == l)]
            put(f"nPos{tk}{lk}", str(int((v.median_diff > 0).sum())), "R2 models with median Grad-Eigen > 0")
            put(f"nNeg{tk}{lk}", str(int((v.median_diff < 0).sum())), "R2 models with median Grad-Eigen < 0")
    a = r2[r2.type == "all"]
    lo, hi = rng(a.median_diff, 3); put("dMedianMin", lo, "R2 all"); put("dMedianMax", hi, "R2 all")
    # dependence on the classifier: condition pattern and same-recipe retraining
    r13 = csv("R13_condition_pattern_by_model")
    if r13 is not None:
        from itertools import combinations
        rho = [r13[[a_, b_]].corr("spearman").iloc[0, 1] for a_, b_ in combinations([m for m in ORDER if m in r13], 2)]
        lo, hi = rng(rho); put("rhoPatternMin", lo, "R13 Spearman between models (18 method-condition means)"); put("rhoPatternMax", hi, "R13")
    for m, mk in METH.items():
        a_ = r3[(r3.model == "A") & (r3.method == m) & (r3.type != "pooled")].mean_xfi.mean()
        b_ = r3[(r3.model == "Arep") & (r3.method == m) & (r3.type != "pooled")].mean_xfi.mean()
        put(f"retrainPct{mk}", num(100 * (b_ - a_) / a_, 0), "R3 (A-rep - A)/A, mean of 9 conditions")
        if (r3.model == "Anorm").any():
            n_ = r3[(r3.model == "Anorm") & (r3.method == m) & (r3.type != "pooled")].mean_xfi.mean()
            put(f"normPct{mk}", num(100 * (n_ - b_) / b_, 0), "R3 (A-norm - A-rep)/A-rep, mean of 9 conditions")
    # Table 3: summary per classifier
    rows = []
    for M in models:
        rows.append(f"{MNAME[M]} & \\xfi{M}GradAll & \\xfi{M}EigenAll & \\cross{M}Grad & \\cross{M}Eigen & \\dMedian{M} & \\pAll{M} & \\rAll{M} & "
                    f"\\nGradLower{M}/\\nEigenLower{M}/\\nNS{M} \\\\")
    TABLES["T3_xfi_summary"] = ("\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Mean XFI} & \\multicolumn{2}{c}{Cross-patient XFI} & \\multicolumn{4}{c}{Grad-CAM $-$ Eigen-CAM (patient level)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-9}\n"
        "Model & Grad & Eigen & Grad & Eigen & Median [95\\% CI] & $p$ & $r$ & G/E/ns \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    # Table S2: condition-wise medians and Holm p for all models
    rows = []
    for t, tk in TYPE.items():
        for l, lk in LEV.items():
            rows.append(f"{TNAME[t]} & L{l} & " + " & ".join(f"\\dMed{M}{tk}{lk} & \\pHolm{M}{tk}{lk}" for M in models) + r" \\")
    TABLES["TS2_grad_vs_eigen_conditions"] = ("\\begin{tabular}{@{}ll" + "rr" * len(models) + "@{}}\n\\toprule\n & & "
        + " & ".join(f"\\multicolumn{{2}}{{c}}{{Model {MNAME[M]}}}" for M in models) + " \\\\\n"
        + "".join(f"\\cmidrule(lr){{{3 + 2 * k}-{4 + 2 * k}}}" for k in range(len(models))) + "\n"
        + "Perturbation & Level & " + " & ".join("Median & $p_{\\mathrm{Holm}}$" for _ in models) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------------------ prediction flips and collapse (Results 3.5)
def sec_flips():
    r12 = csv("R12_flips")
    if r12 is None:
        return missing("flipAGaussThree", "R12")
    fr = r12[r12.kind == "flip_rate"].copy(); fr["level"] = fr.level.astype(int)
    for _, x in fr.iterrows():
        put(f"flip{x.model}{TYPE[x.type]}{LEV[x.level]}", num(100 * x.value, 1), "R12 flip rate (%)")
    for _, x in r12[r12.kind != "flip_rate"].iterrows():
        mk = METH[x.kind.replace("xfi_flip_vs_noflip_", "")]
        put(f"xfiFlip{x.model}{mk}", num(x.xfi_flip), "R12"); put(f"xfiNoFlip{x.model}{mk}", num(x.xfi_noflip), "R12")
    for M in ORDER:
        d = results(M, ["Patient", "Method", "Type", "Level", "pred_clean", "pred_noisy", "prob_noisy"])
        if d is None:
            continue
        d = d[d.Method == "GradCAM"]
        put(f"posClean{M}", num(100 * d.drop_duplicates("Patient").pred_clean.mean(), 1), "results_main share predicted positive, clean")
        for l, lk in LEV.items():
            g = d[(d.Type == "Gaussian") & (d.Level == l)]
            put(f"posGauss{lk}{M}", num(100 * g.pred_noisy.mean(), 1), "results_main share predicted positive, Gaussian")


# ------------------------------------------------------------------ localization and background activation (Results 3.6-3.7)
def sec_localization():
    r5 = csv("R5_clean_iou_by_correctness"); r6 = csv("R6_tp_iou_under_perturbation"); r7 = csv("R7_absolute_false_activation")
    if r5 is None:
        return missing("iouATPGrad", "R5")
    rows = []
    for M in ORDER:
        q = r5[(r5.model == M) & (r5.iou == "Rel-90")]
        if q.empty:
            continue
        for m, mk in METH.items():
            for g in ["TP", "FN", "all positives"]:
                x = q[(q.method == m) & (q.group == g)].iloc[0]
                gk = {"TP": "TP", "FN": "FN", "all positives": "Pos"}[g]
                put(f"iou{M}{gk}{mk}", num(x["mean"]), f"R5 {M} {m} {g} Rel-90"); put(f"iou{M}{gk}{mk}CI", ci(x["mean"], x.ci_low, x.ci_high), "R5")
                if mk == "Grad":
                    put(f"n{M}{gk}", big(x.n), "R5 n")
        rows.append(f"{MNAME[M]} & \\n{M}TP & \\iou{M}TPGradCI & \\iou{M}TPEigenCI & \\n{M}FN & \\iou{M}FNGradCI & \\iou{M}FNEigenCI \\\\")
    TABLES["T4_localization"] = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{True positives} & \\multicolumn{3}{c}{False negatives} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\nModel & $n$ & Grad-CAM & Eigen-CAM & $n$ & Grad-CAM & Eigen-CAM \\\\\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n")
    if r6 is not None:
        for _, x in r6.iterrows():
            k = f"{x.model}{METH[x.method]}{TYPE[x.type]}{LEV[int(x.level)]}"
            put(f"iouN{k}", num(x.noisy), "R6 TP noisy Rel-90"); put(f"drop{k}", num(100 * x.rel_drop, 0), "R6 relative drop (%)")
    if r7 is not None:
        for _, x in r7.iterrows():
            lv = int(x.level)
            if lv == 0 and x.type != "Gaussian":
                continue
            k = f"{x.model}{METH[x.method]}" + ("Clean" if lv == 0 else f"{TYPE[x.type]}{LEV[lv]}")
            put(f"bg{k}", num(x.far_bg_abs5_pos), "R7 background share >=0.5, positives")
            put(f"eout{k}", num(x.energy_out_box_pos), "R7 energy outside box, positives")
            put(f"area{k}", num(x.act_area_abs5_neg), "R7 activated area >=0.5, negatives")


# ------------------------------------------------------------------ validity, baselines, sign, drift (Results 3.8-3.9)
def sec_validity():
    r8 = csv("R8_validity"); r11 = csv("R11_baselines"); r10 = csv("R10_eigen_sign"); r9 = csv("R9_component_ablation")
    if r8 is not None:
        for col, key in [("rho_input_psnr", "Psnr"), ("rho_input_ssim", "Ssim"), ("rho_abs_d_iou_pos", "DIoU"), ("r_flip", "Flip"), ("rho_clean_iou_pos", "CleanIoU")]:
            lo, hi = rng(r8[col]); put(f"rho{key}Min", lo, f"R8 {col}"); put(f"rho{key}Max", hi, f"R8 {col}")
    if r11 is not None:
        b = r11.set_index(["model", "baseline"]).mean_xfi
        put("basePixel", num(b.xs("Random_Pixel", level=1).mean()), "R11"); put("baseSmooth", num(b.xs("Random_Smooth", level=1).mean()), "R11")
        cp = r11[r11.baseline.str.startswith("CrossPatient")].mean_xfi; lo, hi = rng(cp); put("crossMin", lo, "R11"); put("crossMax", hi, "R11")
        mr = r11[r11.baseline.str.startswith("ModelRandom")].mean_xfi; lo, hi = rng(mr); put("randMin", lo, "R11"); put("randMax", hi, "R11")
    if r10 is not None:
        for M in ORDER:
            q = r10[r10.model == M]
            if len(q):
                put(f"minorClean{M}", num(100 * q.minor_lobe_clean.iloc[0], 1), "R10 share of clean Eigen-CAM maps with minor lobe (%)")
                lo, hi = rng(100 * q.sign_flip, 1); put(f"signFlipMin{M}", lo, "R10"); put(f"signFlipMax{M}", hi, "R10")
                put(f"xfiEigenSC{M}", num(q.xfi_eigen_sc.mean()), "R10 mean of 9 conditions, sign-corrected")
    if r9 is not None:
        for c in ["ssim", "pearson", "spearman"]:
            lo, hi = rng(r9[f"rho_xfi_c_{c}"]); put(f"rhoComp{c.capitalize()}Min", lo, "R9"); put(f"rhoComp{c.capitalize()}Max", hi, "R9")
    for M in ["A", "B"]:
        d = results(M, ["Method", "Type", "Level", "com_drift_px"])
        if d is None:
            continue
        for m, mk in METH.items():
            for t, tk in TYPE.items():
                v = d[(d.Method == m) & (d.Type == t) & (d.Level == 3)].com_drift_px.dropna()
                put(f"drift{M}{mk}{tk}", num(v.median(), 1), "results_main median centre-of-mass shift at L3 (px)")


# ------------------------------------------------------------------ supplementary tables S3-S5
TERM = {"Intercept": "Intercept (Eigen-CAM, Gaussian noise, L1)", "Method[T.GradCAM]": "Grad-CAM",
        "Type[T.Blur]": "Blur", "Type[T.SaltPepper]": "Salt-and-pepper", "Level[T.2]": "L2", "Level[T.3]": "L3"}


def term_label(t):
    return " $\\times$ ".join(TERM.get(p, TERM.get(p.replace("Level[T.", "Level[T."), p)) for p in t.split(":"))


def sec_supplement():
    r2b = csv("R2b_mixed_model")
    if r2b is not None:
        models = [m for m in ORDER if m in set(r2b.model)]
        fx = r2b[r2b.ci_low.notna()]
        terms = list(dict.fromkeys(fx.term))
        rows, prev = [], 0
        for t in terms:
            cells = []
            for M in models:
                q = fx[(fx.model == M) & (fx.term == t)]
                cells.append(ci(q.estimate.iloc[0], q.ci_low.iloc[0], q.ci_high.iloc[0]) if len(q) else "--")
            order = t.count(":")
            if rows and order != prev:
                rows.append("\\addlinespace")
            prev = order
            rows.append(term_label(t) + " & " + " & ".join(cells) + r" \\")
        var = []
        for lab, key in [("Patient variance", "random intercept variance (patient)"), ("Residual variance", "residual variance")]:
            var.append(lab + " & " + " & ".join(num(r2b[(r2b.model == M) & (r2b.term == key)].estimate.iloc[0], 4) for M in models) + r" \\")
        gr = r2b[r2b.term.str.startswith("optimiser")]
        var.append("GEE working correlation & " + " & ".join(num(gr[gr.model == M].estimate.iloc[0], 3) for M in models) + r" \\")
        # agreement between LMM and GEE (reported in the caption)
        d = (fx.estimate - fx.gee_estimate).abs().max()
        sig_l, sig_g = fx.p < 0.05, fx.gee_p < 0.05
        width = ((fx.gee_ci_high - fx.gee_ci_low) / (fx.ci_high - fx.ci_low)).replace([np.inf, -np.inf], np.nan)
        mant, ex = f"{d:.1e}".split("e"); put("gldMaxDiff", f"${mant} \\times 10^{{{int(ex)}}}$", "R2b max |LMM - GEE| fixed effect")
        put("glWidthMax", num(width.max(), 1), "R2b max ratio GEE/LMM CI width")
        nch = int((sig_l != sig_g).sum())
        put("glSigText", "the same significance at $\\alpha = 0.05$ for all of them" if nch == 0 else f"the same significance at $\\alpha = 0.05$ for all but {nch} of them",
            "R2b fixed effects whose significance at 0.05 differs between LMM and GEE")
        put("glNTerms", str(len(fx)), "R2b number of fixed-effect estimates")
        TABLES["TS3_mixed_model"] = ("\\begin{tabular}{@{}l" + "c" * len(models) + "@{}}\n\\toprule\nTerm & "
            + " & ".join(f"Model {MNAME[M]}" for M in models) + " \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\midrule\n" + "\n".join(var) + "\n\\bottomrule\n\\end{tabular}\n")
    r4 = csv("R4_slopes")
    if r4 is not None:
        rows = []
        tn = dict(TNAME, pooled="All types")
        for M in [m for m in ORDER if m in set(r4.model)]:
            first = True
            for t in ["Gaussian", "Blur", "SaltPepper", "pooled"]:
                q = r4[(r4.model == M) & (r4.type == t)]
                if not len(q):
                    continue
                q = q.iloc[0]
                rows.append((f"{MNAME[M]}" if first else "") + f" & {tn[t]} & " + ci(q.GradCAM_slope, q.GradCAM_ci_low, q.GradCAM_ci_high)
                            + " & " + ci(q.EigenCAM_slope, q.EigenCAM_ci_low, q.EigenCAM_ci_high) + " & " + ci(q["diff"], q.diff_ci_low, q.diff_ci_high)
                            + f" & {num(q.r, 2)} & {pval(q.p_holm)} \\\\")
                first = False
            rows.append("\\addlinespace")
        TABLES["TS4_slopes"] = ("\\begin{tabular}{@{}llccccc@{}}\n\\toprule\nModel & Perturbation & Grad-CAM slope & Eigen-CAM slope & Difference & $r$ & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    r9 = csv("R9_component_ablation")
    if r9 is not None:
        rows = []
        for M in [m for m in ORDER if m in set(r9.model)]:
            q = r9[r9.model == M].iloc[0]
            rows.append(f"{MNAME[M]} & " + " & ".join(num(q[f"rho_xfi_c_{c}"], 3) for c in ["ssim", "pearson", "spearman"])
                        + " & " + " & ".join(f"{int(q[f'agree_c_{c}_of_9'])}/9" for c in ["ssim", "pearson", "spearman"]) + r" \\")
        TABLES["TS5_component_ablation"] = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{Spearman $\\rho$ with XFI} & \\multicolumn{3}{c}{Same conclusion as XFI} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\nModel & SSIM & Pearson & Spearman & SSIM & Pearson & Spearman \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------------------ sensitivity analyses (05_sensitivity: R14-R17)
def sec_sensitivity():
    r14 = csv("R14_dicom_headers")
    if r14 is not None:
        g = lambda a: r14[r14.attribute == a].set_index("value").n
        pi = g("PhotometricInterpretation"); n = int(pi.sum())
        put("dicomN", big(n), "R14 number of DICOM headers read")
        put("dicomMonoTwo", big(pi.get("MONOCHROME2", 0)), "R14 MONOCHROME2 images")
        put("dicomMonoOne", big(pi.get("MONOCHROME1", 0)), "R14 MONOCHROME1 images")
        bits = g("BitsStored"); put("dicomBits", ", ".join(f"{k} bits ({big(v)})" for k, v in bits.items()) if len(bits) > 1 else f"{bits.index[0]} bits", "R14 BitsStored")
        def present(a):
            q = g(a); return int(q[q.index != "absent"].sum())
        put("dicomVOI", big(present("WindowCenter") + present("VOILUTSequence")), "R14 images with a window or VOI LUT")
        put("dicomRescale", big(present("RescaleSlope") + present("ModalityLUTSequence")), "R14 images with a rescale or modality LUT")
        put("dicomPad", big(present("PixelPaddingValue")), "R14 images with a pixel padding value")
        ts = g("TransferSyntax"); put("dicomTS", ", ".join(ts.index) if len(ts) else "--", "R14 transfer syntax")
    r15 = csv("R15_crosspatient_by_class")
    if r15 is not None:
        models = [m for m in ORDER if m in set(r15.model)]
        rows = []
        for M in models:
            for m in METHODS:
                q = r15[(r15.model == M) & (r15.method == m)].set_index("pairs")
                cells = [ci(q.loc[k].mean_xfi, q.loc[k].ci_low, q.loc[k].ci_high) + f" ({big(q.loc[k].n)})" for k in ["all", "same class", "different class"]]
                rows.append(f"{MNAME[M] if m == 'GradCAM' else ''} & {'Grad-CAM' if m == 'GradCAM' else 'Eigen-CAM'} & " + " & ".join(cells) + r" \\")
                put(f"crossSame{M}{METH[m]}", num(q.loc["same class"].mean_xfi), "R15 cross-patient XFI, same predicted class")
                put(f"crossDiff{M}{METH[m]}", num(q.loc["different class"].mean_xfi), "R15 cross-patient XFI, different predicted class")
            rows.append("\\addlinespace")
        w = r15.pivot_table(index=["model", "method"], columns="pairs", values="mean_xfi")
        for m in METHODS:
            ww = w.xs(m, level="method")
            put(f"nSameLower{METH[m]}", str(int((ww["same class"] < ww["different class"]).sum())), "R15 classifiers with same-class < different-class")
            lo, hi = rng(ww["same class"]); put(f"crossSame{METH[m]}Min", lo, "R15"); put(f"crossSame{METH[m]}Max", hi, "R15")
            lo, hi = rng(ww["different class"]); put(f"crossDiff{METH[m]}Min", lo, "R15"); put(f"crossDiff{METH[m]}Max", hi, "R15")
        put("nModelsSens", str(len(models)), "R15 number of classifiers")
        TABLES["TS6_crosspatient_by_class"] = ("\\begin{tabular}{@{}llccc@{}}\n\\toprule\nModel & Method & All pairs & Same predicted class & Different predicted class \\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    r16 = csv("R16_grad_vs_eigen_noflip")
    if r16 is not None:
        models = [m for m in ORDER if m in set(r16.model)]
        sg = lambda q: 0 if (pd.isna(q.p_holm) or q.p_holm >= 0.05) else int(np.sign(q.median_diff))
        rows, changes, total = [], [], 0
        for t, tk in TYPE.items():
            for l, lk in LEV.items():
                cells = []
                for M in models:
                    a = r16[(r16.model == M) & (r16.type == t) & (r16.level == l) & (r16.subset == "all")].iloc[0]
                    b = r16[(r16.model == M) & (r16.type == t) & (r16.level == l) & (r16.subset == "no flip")].iloc[0]
                    total += 1
                    if sg(a) != sg(b):
                        changes.append(f"Model {MNAME[M]}, " + (TNAME[t] if t != "SaltPepper" else "salt-and-pepper") + f" L{l}")
                    star = "" if sg(b) == 0 else "*"
                    cells.append((num(b.median_diff) + star if not pd.isna(b.median_diff) else "--") + f" & {100 * b.share_kept:.0f}")
                rows.append(f"{TNAME[t]} & L{l} & " + " & ".join(cells) + r" \\")
        put("noflipChange", str(len(changes)), "R16 model-conditions whose significant direction changes without flips")
        for M in models:
            for l, lk in LEV.items():
                q = r16[(r16.model == M) & (r16.type == "Gaussian") & (r16.level == l) & (r16.subset == "no flip")]
                if len(q): put(f"kept{M}Gauss{lk}", f"{100 * q.share_kept.iloc[0]:.0f}", "R16 share of patients without flip (%)")
        put("noflipTotal", str(total), "R16 number of model-conditions")
        put("noflipChangeList", "; ".join(changes) if changes else "none", "R16 list of changed model-conditions")
        TABLES["TS7_grad_vs_eigen_noflip"] = ("\\begin{tabular}{@{}ll" + "rr" * len(models) + "@{}}\n\\toprule\n & & "
            + " & ".join(f"\\multicolumn{{2}}{{c}}{{Model {MNAME[M]}}}" for M in models) + " \\\\\n"
            + "".join(f"\\cmidrule(lr){{{3 + 2 * k}-{4 + 2 * k}}}" for k in range(len(models))) + "\n"
            + "Perturbation & Level & " + " & ".join("Median & \\%" for _ in models) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    r17 = csv("R17_validity_ci")
    if r17 is not None:
        hw = np.maximum(r17.rho - r17.ci_low, r17.ci_high - r17.rho)
        put("rhoCIHalfMax", num(hw.max(), 3), "R17 largest half-width of the cluster-bootstrap CI")
        put("rhoCIAllExcludeZero", "all" if bool(((r17.ci_low > 0) | (r17.ci_high < 0)).all()) else "not all", "R17 CIs exclude zero")
    r18 = csv("R18_xfi_by_outcome")
    if r18 is not None:
        models = [m for m in ORDER if m in set(r18.model)]
        GN = {"TP": "true positives", "FN": "false negatives", "TN": "true negatives", "FP": "false positives"}
        def join(xs):
            return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
        rows, phrases = [], []
        for M in models:
            q = r18[r18.model == M].set_index("group")
            ntot = q.n.sum()
            sign = {g: (0 if q.loc[g].p_holm >= 0.05 else int(np.sign(q.loc[g].median_diff))) for g in q.index}
            first = True
            for g in [g for g in ["TP", "FN", "TN", "FP"] if g in q.index]:
                r = q.loc[g]
                put(f"xfiOut{M}{g}Grad", num(r.grad_mean), "R18 mean patient-level XFI by clean outcome")
                put(f"xfiOut{M}{g}Eigen", num(r.eigen_mean), "R18 mean patient-level XFI by clean outcome")
                put(f"nOut{M}{g}", big(r.n), "R18 patients per outcome group")
                put(f"pctOut{M}{g}", f"{100 * r.n / ntot:.0f}", "R18 share of test patients per outcome group (%)")
                rows.append((f"{MNAME[M]}" if first else "") + f" & {g} & {big(r.n)} & " + ci(r.grad_mean, r.grad_ci_low, r.grad_ci_high)
                            + " & " + ci(r.eigen_mean, r.eigen_ci_low, r.eigen_ci_high) + " & " + ci(r.median_diff, r.median_ci_low, r.median_ci_high)
                            + f" & {pval(r.p_holm)} \\\\")
                first = False
            rows.append("\\addlinespace")
            vals = set(sign.values())
            word = {1: "more fragile", -1: "less fragile", 0: "not significantly different"}
            if len(vals) == 1:
                ph = f"{word[vals.pop()]} in all four groups in Model {MNAME[M]}"
            else:
                ph = " and ".join(word[v] + " for " + join([GN[g] for g in ["TP", "FN", "TN", "FP"] if sign.get(g) == v]) for v in [1, -1, 0] if v in vals) + f" in Model {MNAME[M]}"
            phrases.append(ph); put(f"outSign{M}", ph, "R18 sign of Grad - Eigen by outcome group")
        put("outSignAll", "; ".join(phrases), "R18 sign of Grad - Eigen by outcome group, all classifiers")
        TABLES["TS8_xfi_by_outcome"] = ("\\begin{tabular}{@{}llrcccc@{}}\n\\toprule\nModel & Outcome & $n$ & Grad-CAM & Eigen-CAM & Grad $-$ Eigen (median) & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ main
def main():
    sec_data(); sec_classifiers(); sec_xfi(); sec_flips(); sec_localization(); sec_validity(); sec_supplement(); sec_sensitivity()
    OUT.mkdir(exist_ok=True)
    lines = ["% generated by scripts/make_numbers_tex.py, do not edit by hand"]
    seen = set()
    for n, v, s in ROWS:
        assert n not in seen, f"duplicate macro {n}"; seen.add(n)
        lines.append(f"\\newcommand{{\\{n}}}{{{v}\\xspace}}  % {s}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(ROWS, columns=["macro", "value", "source"]).to_csv(OUT / "numbers.csv", index=False)
    (OUT / "tables").mkdir(exist_ok=True)
    for k, v in TABLES.items():
        (OUT / "tables" / f"{k}.tex").write_text("% generated by scripts/make_numbers_tex.py\n" + v, encoding="utf-8")
    miss = [n for n, _, s in ROWS if s.startswith("MISSING")]
    print(f"{len(ROWS)} macros -> manuscript/numbers.tex, {len(TABLES)} tables -> manuscript/tables/" + (f" | MISSING: {miss}" if miss else ""))


if __name__ == "__main__":
    main()
