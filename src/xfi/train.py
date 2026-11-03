"""Training of the additional classifiers of the R1 revision (D-07, D-08, D-29).

    python -m xfi.train --model Arep     # v1 recipe, no ImageNet normalisation (replica, for R1-d)
    python -m xfi.train --model Anorm    # v1 recipe + ImageNet normalisation (R1-d)
    python -m xfi.train --model B        # ImageNet-pretrained, pos_weight, val-AUC early stopping (R1-b, R1-V2, R2-M3)

All three train on train_fit (v1 training set minus a stratified 10% validation split), log validation
loss, AUC, ECE and Brier after every epoch, and finally predict the validation and the frozen test set.
Outputs: model/<name>_best.pth, output/phase2/train_<name>/{epochs.csv, val_probs.csv, test_probs.csv, summary.json, train.log}
"""
import argparse, json, os, random, sys, time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xfi import data as D

SPECS = {
    "Arep":  dict(pretrained=False, imagenet_norm=False, epochs=5,  batch=16, lr=1e-4, pos_weight=False, select="train_loss", patience=None),
    "Anorm": dict(pretrained=False, imagenet_norm=True,  epochs=5,  batch=16, lr=1e-4, pos_weight=False, select="train_loss", patience=None),
    "B":     dict(pretrained=True,  imagenet_norm=True,  epochs=20, batch=32, lr=1e-4, pos_weight=True,  select="val_auc",    patience=4),
}


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


def ece(y, p, n_bins=10):
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in range(n_bins) if (b == k).any()))


def build_model(pretrained):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); ps, ys, ids = [], [], []
    for x, y, pid in loader:
        ps.append(torch.sigmoid(model(x.to(device, non_blocking=True))).squeeze(1).float().cpu().numpy())
        ys.append(y.numpy()); ids += list(pid)
    return np.concatenate(ps), np.concatenate(ys), ids


def evaluate(y, p):
    eps = 1e-7; pc = np.clip(p, eps, 1 - eps)
    return dict(loss=float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), auc=float(roc_auc_score(y, p)),
                ece=ece(y, p), brier=float(np.mean((p - y) ** 2)))


def threshold_metrics(y, p, thr):
    yh = (p > thr).astype(int); y = y.astype(int)
    tp, fp = int(((yh == 1) & (y == 1)).sum()), int(((yh == 1) & (y == 0)).sum())
    tn, fn = int(((yh == 0) & (y == 0)).sum()), int(((yh == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan"); rec = tp / (tp + fn); spec = tn / (tn + fp)
    return dict(threshold=float(thr), TP=tp, FP=fp, TN=tn, FN=fn, accuracy=(tp + tn) / len(y), precision=prec, recall=rec,
                specificity=spec, f1=2 * prec * rec / (prec + rec) if prec + rec else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS))
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N patients per split")
    ap.add_argument("--epochs", type=int, default=0, help="debug: override number of epochs")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    spec = dict(SPECS[a.model]);
    if a.epochs: spec["epochs"] = a.epochs
    cfg = D.load_cfg(); seed = cfg["project"]["seed"]
    out = D.ROOT / "output" / "phase2" / f"train_{a.model}"; out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} [{a.model}] {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"spec={spec} device={device} torch={torch.__version__}")
    D.build_cache(cfg, log=log)
    sp = D.load_splits(cfg, seed=seed)
    if a.limit:
        sp = {k: v.groupby("Target", group_keys=False).apply(lambda g: g.head(max(2, a.limit // 2))) for k, v in sp.items()}
    log(" ".join(f"{k}={len(v)}(pos {int(v.Target.sum())})" for k, v in sp.items()))
    cache = D.ImageCache(cfg)
    g = torch.Generator(); g.manual_seed(seed)
    mk = lambda df, train, bs, shuffle, nw, persist: DataLoader(D.CXRDataset(df, cache, train, spec["imagenet_norm"]), batch_size=bs,
                                                   shuffle=shuffle, num_workers=nw, pin_memory=True, worker_init_fn=worker_init,
                                                   generator=g, persistent_workers=persist and nw > 0)
    tr = mk(sp["train_fit"], True, spec["batch"], True, a.workers, True)
    ew = min(4, a.workers)
    va = mk(sp["val"], False, 64, False, ew, False); te = mk(sp["test"], False, 64, False, ew, False)

    model = build_model(spec["pretrained"]).to(device)
    n_pos = float(sp["train_fit"].Target.sum()); n_neg = float(len(sp["train_fit"]) - n_pos)
    pw = torch.tensor([n_neg / n_pos], device=device) if spec["pos_weight"] else None
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
    log(f"pos_weight={None if pw is None else round(pw.item(), 4)}")

    best_val, best_ep, since, rows = None, None, 0, []
    f_best = D.ROOT / "model" / f"{a.model}_best.pth"
    for ep in range(1, spec["epochs"] + 1):
        model.train(); t0 = time.time(); tot, n = 0.0, 0
        for i, (x, y, _) in enumerate(tr):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).unsqueeze(1)
            opt.zero_grad(set_to_none=True); loss = crit(model(x), y); loss.backward(); opt.step()
            tot += loss.item() * len(x); n += len(x)
            if (i + 1) % 200 == 0:
                log(f"epoch {ep} batch {i + 1}/{len(tr)} loss {tot / n:.4f}")
        pv, yv, _ = predict(model, va, device); ev = evaluate(yv, pv)
        row = dict(epoch=ep, train_loss=tot / n, val_loss=ev["loss"], val_auc=ev["auc"], val_ece=ev["ece"], val_brier=ev["brier"],
                   seconds=round(time.time() - t0, 1)); rows.append(row)
        pd.DataFrame(rows).to_csv(out / "epochs.csv", index=False)
        crit_val = -row["train_loss"] if spec["select"] == "train_loss" else row["val_auc"]
        improved = best_val is None or crit_val > best_val
        if improved:
            best_val, best_ep, since = crit_val, ep, 0; torch.save(model.state_dict(), f_best)
        else:
            since += 1
        if spec["select"] == "train_loss":
            torch.save(model.state_dict(), D.ROOT / "model" / f"{a.model}_epoch{ep}.pth")
        log(f"epoch {ep} done: " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch") + (" *best*" if improved else ""))
        if spec["patience"] and since >= spec["patience"]:
            log(f"early stopping (no val AUC improvement for {since} epochs)"); break

    model.load_state_dict(torch.load(f_best, map_location=device, weights_only=True))
    pv, yv, iv = predict(model, va, device); pt, yt, it = predict(model, te, device)
    pd.DataFrame({"patientId": iv, "Target": yv, "prob": pv}).to_csv(out / "val_probs.csv", index=False)
    pd.DataFrame({"patientId": it, "Target": yt, "prob": pt}).to_csv(out / "test_probs.csv", index=False)
    fpr, tpr, th = roc_curve(yv, pv); thr_val = float(th[np.argmax(tpr - fpr)])
    summary = dict(model=a.model, spec=spec, best_epoch=best_ep, selection=spec["select"], weights=str(f_best.relative_to(D.ROOT)),
                   weights_sha256=D.sha256(f_best), n=dict((k, len(v)) for k, v in sp.items()), pos_weight=None if pw is None else pw.item(),
                   val=evaluate(yv, pv), test=evaluate(yt, pt), val_youden_threshold=thr_val,
                   test_at_0p5=threshold_metrics(yt, pt, 0.5), test_at_val_threshold=threshold_metrics(yt, pt, thr_val),
                   torch=torch.__version__, finished=f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"FINISHED best_epoch={best_ep} test_auc={summary['test']['auc']:.4f} test_ece={summary['test']['ece']:.4f} "
        f"val_thr={thr_val:.4f} recall@valthr={summary['test_at_val_threshold']['recall']:.4f}")


if __name__ == "__main__":
    main()
