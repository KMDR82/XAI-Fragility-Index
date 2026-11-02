"""Phase 0 diagnostic: compare the Kaggle "last version" outputs (weights + CSV) with
the Drive weights (ART-002), the v1 legacy CSV (ART-001) and the numbers in the submitted manuscript.

Usage (repo root, env xfi):
    python scripts/p0_compare_kaggle_outputs.py                     # auto-detects output/*.pth and output/*.csv
    python scripts/p0_compare_kaggle_outputs.py --pth X.pth --csv Y.csv
    python scripts/p0_compare_kaggle_outputs.py --no-infer          # skip the GPU inference part

Writes nothing except output/phase0/kaggle_compare_report.txt and output/phase0/kaggleA_test_probs.csv.
"""
import argparse, hashlib, sys, glob, io, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon, spearmanr

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
P = {k: ROOT / v for k, v in CFG["paths"].items() if k != "drive_backup_dir"}

# Numbers printed in the submitted manuscript (Table 1, abstract, Results)
PAPER = {"mean_xfi": {"GradCAM": 0.2072, "EigenCAM": 0.2694},
         "ci": {"GradCAM": 0.0021, "EigenCAM": 0.0023},
         "slope": {"GradCAM": -0.0741, "EigenCAM": -0.0580},
         "clean_iou90": {"GradCAM": 0.1066, "EigenCAM": 0.0566},
         "far": {"GradCAM": 0.0977, "EigenCAM": 0.0998},
         "r_xfi": 0.2552, "r_ssim": 0.3535,
         "random": 0.6649, "uniform": 0.6667}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def report_numbers(df, name):
    """Recompute the manuscript-level numbers exactly as the repo's generate_q1_report() does."""
    print(f"\n--- manuscript-level numbers recomputed from {name} ---")
    out = {}
    xai = [m for m in ["GradCAM", "EigenCAM"] if m in set(df["Method"])]
    for metric in ["ssim", "xfi"]:
        if metric not in df.columns or len(xai) < 2:
            continue
        pv = df.pivot_table(index=["Patient", "Level", "Type"], columns="Method", values=metric).dropna()
        stat, p = wilcoxon(pv["GradCAM"], pv["EigenCAM"])
        n = len(pv); mu = n * (n + 1) / 4; sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        r = abs((stat - mu) / sd) / np.sqrt(n)
        out[f"r_{metric}"] = r
        print(f"  Wilcoxon {metric:5s}: n={n}  p={p:.3e}  r={r:.4f}   (paper r={PAPER['r_' + metric]})")
    print(f"  {'method':17s} {'meanXFI':>8s} {'CI95':>7s} {'slope':>8s} {'IoU90c':>7s} {'IoU90n':>7s} {'FAR':>7s}")
    for m in df["Method"].unique():
        s = df[df["Method"] == m]
        lv = sorted(s["Level"].unique()); y = s.groupby("Level")["xfi"].mean().loc[lv].values
        slope = np.polyfit(lv, y, 1)[0] if len(lv) > 1 else np.nan
        ci = 1.96 * s["xfi"].std() / np.sqrt(len(s))
        c = s["iou_c_rel_90"].mean() if "iou_c_rel_90" in s else np.nan
        nn = s["iou_n_rel_90"].mean() if "iou_n_rel_90" in s else np.nan
        far = s["false_activation"].mean() if "false_activation" in s else np.nan
        out[m] = dict(mean=s["xfi"].mean(), ci=ci, slope=slope, iou_c=c, iou_n=nn, far=far)
        print(f"  {m:17s} {s['xfi'].mean():8.4f} {ci:7.4f} {slope:8.4f} {c:7.4f} {nn:7.4f} {far:7.4f}")
    print("  paper:            GradCAM 0.2072/0.0021/-0.0741/0.1066/-/0.0977 | EigenCAM 0.2694/0.0023/-0.0580/0.0566/-/0.0998")
    return out


def describe_csv(df, name):
    pats = list(dict.fromkeys(df["Patient"]))
    inc = np.array([a < b for a, b in zip(pats, pats[1:])])
    print(f"{name}: rows={len(df)}  patients={len(pats)}  positives={int(df.drop_duplicates('Patient')['Label'].sum())}")
    print(f"  methods={sorted(df['Method'].unique())}")
    print(f"  types={sorted(df['Type'].unique())}  levels={sorted(df['Level'].unique())}")
    print(f"  columns={list(df.columns)}")
    print(f"  patient order: sorted-run breaks={(~inc).sum()} at {np.where(~inc)[0][:5].tolist()}")
    return pats


def infer(pth, ids):
    import torch, torch.nn as nn, torchvision.models as models, torchvision.transforms as T, cv2, pydicom
    from torch.utils.data import Dataset, DataLoader
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, 1)
    m.load_state_dict(torch.load(pth, map_location="cpu", weights_only=True), strict=True)
    m = m.to(dev).eval()
    tf = T.Compose([T.ToPILImage(), T.Resize((224, 224)), T.ToTensor()])

    class DS(Dataset):
        def __len__(self): return len(ids)
        def __getitem__(self, i):
            img = pydicom.dcmread(P["image_dir"] / f"{ids[i]}.dcm").pixel_array
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return tf(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)), ids[i]

    probs, pids = [], []
    with torch.no_grad():
        for x, pid in DataLoader(DS(), batch_size=64, num_workers=8):
            probs.append(torch.sigmoid(m(x.to(dev))).squeeze(1).cpu().numpy()); pids += list(pid)
    return pd.Series(np.concatenate(probs), index=pids)


def compare_probs(a, b, name_a, name_b):
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).abs()
    print(f"  {name_a} vs {name_b}: n={len(common)}  max|d|={d.max():.6f}  mean|d|={d.mean():.6f}  "
          f"n(|d|>1e-3)={(d > 1e-3).sum()}  spearman={spearmanr(a.loc[common], b.loc[common])[0]:.4f}")
    return d.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth"); ap.add_argument("--csv"); ap.add_argument("--no-infer", action="store_true")
    ap.add_argument("--v1", default=str(P["v1_legacy_csv"]))
    a = ap.parse_args()

    pth = a.pth or next(iter(sorted(glob.glob(str(P["output_dir"] / "*.pth")))), None)
    csv = a.csv or next(iter(sorted(glob.glob(str(P["output_dir"] / "*.csv")))), None)

    section("1. FILES")
    for label, f in [("Kaggle weights", pth), ("Drive weights (ART-002)", P["model_a_weights"]),
                     ("Kaggle CSV", csv), ("v1 legacy CSV (ART-001)", a.v1)]:
        if f and Path(f).exists():
            print(f"{label:25s} {Path(f).name:45s} {Path(f).stat().st_size:>12,d} B  sha256={sha256(f)}")
        else:
            print(f"{label:25s} NOT FOUND ({f})")
    same_w = pth and Path(pth).exists() and sha256(pth) == sha256(P["model_a_weights"])
    print(f"\nKaggle weights identical to Drive weights: {same_w}")

    v1 = pd.read_csv(a.v1)
    kg = pd.read_csv(csv) if csv and Path(csv).exists() else None

    section("1b. XFI COLUMN")
    if "robustness_score" in v1.columns and "xfi" in v1.columns:
        d = (v1["xfi"] - (1 - v1["robustness_score"])).abs()
        print(f"v1 CSV: max|xfi - (1 - robustness_score)| = {d.max():.2e}  (0 means xfi was appended as 1 - RS)")
    if kg is not None and "xfi" not in kg.columns:
        if "robustness_score" in kg.columns:
            kg["xfi"] = 1 - kg["robustness_score"]
            print("Kaggle CSV has no 'xfi' column -> computed here as 1 - robustness_score")
        else:
            print("Kaggle CSV has neither 'xfi' nor 'robustness_score'")

    section("2. CSV STRUCTURE")
    v1_p = describe_csv(v1, "v1 legacy CSV")
    if kg is not None:
        kg_p = describe_csv(kg, "Kaggle CSV")
        s1, s2 = set(v1_p), set(kg_p)
        print(f"\npatients: common={len(s1 & s2)}  only_v1={len(s1 - s2)}  only_kaggle={len(s2 - s1)}")
        if len(kg_p) <= len(v1_p):
            print(f"Kaggle patient list equals the first {len(kg_p)} v1 patients in order: {kg_p == v1_p[:len(kg_p)]}")

    section("3. MODEL PROBABILITIES")
    v1_prob = v1.drop_duplicates("Patient").set_index("Patient")["Model_Prob"]
    if kg is not None:
        kg_prob = kg.drop_duplicates("Patient").set_index("Patient")["Model_Prob"]
        compare_probs(kg_prob, v1_prob, "Kaggle CSV", "v1 CSV")
    if not a.no_infer and pth and Path(pth).exists():
        ids = list(v1_prob.index)
        if same_w and (P["phase0_out"] / "modelA_test_probs.csv").exists():
            pA = pd.read_csv(P["phase0_out"] / "modelA_test_probs.csv", index_col=0).iloc[:, 0]
            print("  (Kaggle weights == Drive weights; reusing C4 probabilities)")
        else:
            print("  running inference with the Kaggle weights on the 5,337 test images ...")
            pA = infer(pth, ids)
            pA.to_csv(P["phase0_out"] / "kaggleA_test_probs.csv")
        compare_probs(pA, v1_prob, "Kaggle weights", "v1 CSV")
        if kg is not None:
            compare_probs(pA, kg_prob, "Kaggle weights", "Kaggle CSV")

    section("4. METRIC VALUES: Kaggle CSV vs v1 CSV (same patient, method, type, level)")
    if kg is not None:
        key = ["Patient", "Method", "Type", "Level"]
        cols = [c for c in ["robustness_score", "xfi", "ssim", "p_corr", "s_corr", "iou_c_rel_90", "iou_n_rel_90", "false_activation"]
                if c in kg.columns and c in v1.columns]
        mg = v1.merge(kg, on=key, suffixes=("_v1", "_kg"))
        print(f"matched rows: {len(mg)}")
        for c in cols:
            d = (mg[f"{c}_v1"] - mg[f"{c}_kg"]).abs()
            print(f"  {c:18s} max|d|={d.max():.6f}  mean|d|={d.mean():.6f}")

    section("5. WHICH FILE GIVES THE MANUSCRIPT NUMBERS?")
    report_numbers(v1, "v1 legacy CSV")
    if kg is not None:
        report_numbers(kg, "Kaggle CSV")
        print("\n--- robustness score (RS) by method and level, Kaggle CSV ---")
        print(kg[kg.Method.isin(["GradCAM", "EigenCAM"])].groupby(["Method", "Level"])["robustness_score"].mean().unstack().round(4).to_string())


if __name__ == "__main__":
    buf = io.StringIO()
    class Tee:
        def write(self, s): sys.__stdout__.write(s); buf.write(s)
        def flush(self): sys.__stdout__.flush()
    with contextlib.redirect_stdout(Tee()):
        main()
    (P["phase0_out"]).mkdir(parents=True, exist_ok=True)
    (P["phase0_out"] / "kaggle_compare_report.txt").write_text(buf.getvalue())
