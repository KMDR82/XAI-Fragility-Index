"""Statistics used throughout the revision (D-09, D-15). The patient is the unit of resampling."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, norm, rankdata
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score


# ---------------------------------------------------------------- bootstrap
def boot_idx(n, B, seed):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def boot_ci_stat(x, stat=np.mean, B=2000, seed=42, alpha=0.05, idx=None):
    """Point estimate and percentile CI of stat(x) by resampling elements of x (one per patient)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if idx is None or idx.shape[1] != len(x):
        idx = boot_idx(len(x), B, seed)
    bs = stat(x[idx], axis=1)
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    return float(stat(x)), float(lo), float(hi)


def ratio_of_means_ci(num, den, B=2000, seed=42, alpha=0.05):
    """(mean(num) / mean(den)) with a paired patient bootstrap CI."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = boot_idx(len(num), B, seed)
    bs = num[idx].mean(1) / den[idx].mean(1)
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    return float(num.mean() / den.mean()), float(lo), float(hi)


# ---------------------------------------------------------------- classifier
def threshold_metrics(y, p, thr):
    y = np.asarray(y).astype(int); yh = (np.asarray(p) >= thr).astype(int) if thr != 0.5 else (np.asarray(p) > thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); fp = int(((yh == 1) & (y == 0)).sum())
    tn = int(((yh == 0) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    sd = lambda a, b: a / b if b else np.nan
    prec, rec, spec = sd(tp, tp + fp), sd(tp, tp + fn), sd(tn, tn + fp)
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, accuracy=sd(tp + tn, len(y)), precision=prec,
                recall=rec, specificity=spec, f1=sd(2 * prec * rec, prec + rec) if prec == prec and rec == rec else np.nan,
                balanced_accuracy=(rec + spec) / 2)


def ece_score(y, p, n_bins=10):
    """Expected calibration error with equal-width bins on [0, 1]."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def brier(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def youden_threshold(y, p):
    fpr, tpr, th = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return float(th[j])


def classifier_table(y, p, thresholds, B=2000, seed=42, alpha=0.05):
    """Rows: metric; columns: estimate and CI for each threshold. AUC, AP, ECE, Brier are threshold-free."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    idx = boot_idx(len(y), B, seed)
    rows = []

    def add(name, f, thr_label):
        est = f(y, p)
        bs = np.array([f(y[i], p[i]) for i in idx])
        lo, hi = np.nanquantile(bs, [alpha / 2, 1 - alpha / 2])
        rows.append(dict(metric=name, threshold=thr_label, estimate=est, ci_low=lo, ci_high=hi))

    add("AUC", lambda a, b: roc_auc_score(a, b), "threshold-free")
    add("average_precision", lambda a, b: average_precision_score(a, b), "threshold-free")
    add("ECE_10bins", lambda a, b: ece_score(a, b), "threshold-free")
    add("Brier", lambda a, b: brier(a, b), "threshold-free")
    for label, thr in thresholds.items():
        for m in ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy"]:
            add(m, lambda a, b, thr=thr, m=m: threshold_metrics(a, b, thr)[m], label)
    return pd.DataFrame(rows)


def reliability_curve(y, p, n_bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out.append(dict(bin=b, bin_low=b / n_bins, bin_high=(b + 1) / n_bins, n=int(m.sum()),
                            mean_pred=p[m].mean(), frac_pos=y[m].mean()))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- paired tests
def wilcoxon_paired(a, b):
    """Two-sided Wilcoxon signed-rank (zero differences dropped, normal approximation with tie
    correction). Returns n_nonzero, W, p, |z| and effect size r = |z| / sqrt(n_nonzero)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    nz = int((d != 0).sum())
    if nz == 0:
        return dict(n=len(d), n_nonzero=0, W=np.nan, p=1.0, z=0.0, r=0.0)
    res = wilcoxon(a, b, zero_method="wilcox", method="approx")
    # z computed directly from W+ (tie-corrected normal approximation), so it stays finite when p underflows to 0
    dn = d[d != 0]
    ranks = rankdata(np.abs(dn))
    w_plus = ranks[dn > 0].sum()
    _, counts = np.unique(np.abs(dn), return_counts=True)
    var = nz * (nz + 1) * (2 * nz + 1) / 24 - (counts ** 3 - counts).sum() / 48
    z = float(abs(w_plus - nz * (nz + 1) / 4) / np.sqrt(var))
    return dict(n=len(d), n_nonzero=nz, W=float(res.statistic), p=float(res.pvalue), z=z, r=z / np.sqrt(nz))


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i]); adj[i] = min(1.0, running)
    return adj


def per_patient_slope(df, value="xfi", level="Level", by=("Patient", "Method")):
    """OLS slope of value on level for every group in `by` (vectorised)."""
    g = df.groupby(list(by))
    xm = g[level].transform("mean"); ym = g[value].transform("mean")
    t = df.assign(_sxy=(df[level] - xm) * (df[value] - ym), _sxx=(df[level] - xm) ** 2)
    s = t.groupby(list(by))[["_sxy", "_sxx"]].sum()
    return (s["_sxy"] / s["_sxx"]).rename("slope").reset_index()
