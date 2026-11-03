"""Metrics of the v2 run.

XFI (Eq. 1-9, unchanged; D-16):  C, N min-max normalised;  RS = (SSIM + (r_p + 1)/2 + (r_s + 1)/2) / 3;  XFI = 1 - RS.
Conventions kept from v1: SSIM = 0 if either map is constant; an undefined correlation contributes 0.5.
Theoretical range: XFI in [0, 4/3]; two uncorrelated maps give about 2/3.

Localization (D-13, R1-BR1): every IoU is computed against the ground-truth box mask, separately for the clean and the
perturbed heatmap. Activation outside the box (D-03, R2-M2) uses absolute thresholds.
"""
import numpy as np
from scipy.stats import rankdata
from skimage.metrics import structural_similarity

REL = (0.80, 0.90)
ABS = (0.5, 0.7)


def normalize(h):
    h = np.asarray(h, dtype=np.float64)
    return (h - h.min()) / (h.max() - h.min() + 1e-8)


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def xfi(clean, noisy):
    c, n = normalize(clean), normalize(noisy)
    degenerate = int(c.std() < 1e-8 or n.std() < 1e-8)
    s = 0.0 if degenerate else float(structural_similarity(c, n, data_range=1.0))
    cf, nf = c.ravel(), n.ravel()
    rp = _pearson(cf, nf)
    rs = _pearson(rankdata(cf), rankdata(nf)) if not np.isnan(rp) else float("nan")
    rpn = 0.5 if np.isnan(rp) else (rp + 1) / 2
    rsn = 0.5 if np.isnan(rs) else (rs + 1) / 2
    robust = (s + rpn + rsn) / 3
    return dict(ssim=s, p_corr=rp, s_corr=rs, rs=robust, xfi=1.0 - robust, degenerate=degenerate)


def gt_mask(boxes, size=224, orig=1024):
    """Union of the RSNA boxes (x, y, w, h in original pixels) rescaled to size x size (v1 rule)."""
    m = np.zeros((size, size), dtype=bool)
    f = size / orig
    for x, y, w, h in boxes:
        x1, y1 = np.clip(int(x * f), 0, size), np.clip(int(y * f), 0, size)
        x2, y2 = np.clip(int((x + w) * f), 0, size), np.clip(int((y + h) * f), 0, size)
        m[y1:y2, x1:x2] = True
    return m


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float("nan")


def localization(h, mask):
    """IoU with the box at relative and absolute thresholds, energy outside the box, background activation."""
    h = normalize(h); out = {}
    for t in REL:
        out[f"iou_rel{int(t * 100)}"] = _iou(h >= np.percentile(h, t * 100), mask)
    bg = ~mask
    for t in ABS:
        act = h >= t; k = f"{int(t * 10)}"
        out[f"iou_abs{k}"] = _iou(act, mask)
        out[f"far_bg_abs{k}"] = float((act & bg).sum() / bg.sum()) if bg.any() else float("nan")
        out[f"prec_in_box_abs{k}"] = float((act & mask).sum() / act.sum()) if act.any() else float("nan")
    tot = h.sum()
    out["energy_out_box"] = float(h[bg].sum() / tot) if tot > 0 else float("nan")
    return out


def activation_area(h):
    h = normalize(h)
    return {f"act_area_abs{int(t * 10)}": float((h >= t).mean()) for t in ABS}


def center_of_mass(h):
    h = normalize(h); s = h.sum()
    if s <= 0:
        return np.array([np.nan, np.nan])
    yy, xx = np.mgrid[0:h.shape[0], 0:h.shape[1]]
    return np.array([(yy * h).sum() / s, (xx * h).sum() / s])
