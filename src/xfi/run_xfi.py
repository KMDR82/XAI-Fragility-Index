"""Canonical v2 XFI run (Phase 3).

    python -m xfi.run_xfi --model A                  # full test set, Model A (ART-008), threshold 0.5
    python -m xfi.run_xfi --model B                  # Model B, threshold chosen on validation (D-30)
    python -m xfi.run_xfi --model A --limit 50 --out output/phase2/pilot_A   # pilot

Per test patient: clean image + 9 perturbed images (3 types x 3 levels) -> class probabilities, Grad-CAM, Eigen-CAM,
Eigen-CAM with sign correction, and Grad-CAM / Eigen-CAM of a randomised copy of the model on the clean image.
Rows (one per patient x method x condition) go to results_main; baselines to results_base.
Shards of --shard patients are written atomically; an interrupted run resumes at the first missing shard and,
because every random draw is keyed by (seed, patient, condition), reproduces the same numbers.
"""
import argparse, json, os, platform, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xfi import data as D, perturb as P, metrics as M, xai as X

MODELS = {
    "A":     dict(weights="model/modelA_kaggle_v1.pth", imagenet_norm=False, threshold=0.5),
    "B":     dict(weights="model/B_best.pth", imagenet_norm=True, threshold="output/phase2/train_B/summary.json"),
    "Anorm": dict(weights="model/Anorm_best.pth", imagenet_norm=True, threshold="output/phase2/train_Anorm/summary.json"),
    "Arep":  dict(weights="model/Arep_best.pth", imagenet_norm=False, threshold="output/phase2/train_Arep/summary.json"),
}
CAM_METHODS = ("GradCAM", "EigenCAM", "EigenCAM_SC")
RAND_METHODS = ("GradCAM", "EigenCAM")
MEAN = np.array(D.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
STD = np.array(D.IMAGENET_STD, dtype=np.float32)[:, None, None]


def to_input(gray, imagenet_norm):
    """(H, W) in [0,1] -> (3, H, W) float32: the same image in all three channels, optional ImageNet normalisation."""
    x = np.repeat(gray[None], 3, axis=0).astype(np.float32)
    return (x - MEAN) / STD if imagenet_norm else x


def load_model(spec, device):
    m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, 1)
    m.load_state_dict(torch.load(D.ROOT / spec["weights"], map_location="cpu", weights_only=True), strict=True)
    return m.to(device).eval()


def resolve_threshold(spec):
    t = spec["threshold"]
    return float(t) if isinstance(t, (int, float)) else float(json.loads((D.ROOT / t).read_text())["val_youden_threshold"])


# ------------------------------------------------------------------ CPU part (worker processes)
def compute_rows(job):
    pid, label, boxes, clean, noisy, probs, cls, thr, maps, signed, rmaps, seed = (
        job[k] for k in ("pid", "label", "boxes", "clean", "noisy", "probs", "cls", "thr", "maps", "signed", "rmaps", "seed"))
    conds = P.conditions()
    mask = M.gt_mask(boxes) if label == 1 and len(boxes) else None
    iq = [P.input_quality(clean, noisy[j]) for j in range(len(conds))]
    base_info = dict(Patient=pid, Label=label, prob_clean=probs[0], pred_clean=cls, threshold=thr)
    main = []
    for m in CAM_METHODS:
        c = maps[m][0]
        loc_c = M.localization(c, mask) if mask is not None else {}
        area_c = M.activation_area(c); com_c = M.center_of_mass(c)
        for j, (t, l) in enumerate(conds):
            n = maps[m][j + 1]
            r = dict(base_info, Method=m, Type=t, Level=l, prob_noisy=probs[j + 1], pred_noisy=int(probs[j + 1] > thr),
                     input_psnr=iq[j][0], input_ssim=iq[j][1], **M.xfi(c, n))
            r["flip"] = int(r["pred_noisy"] != cls)
            r.update({f"{k}_c": v for k, v in loc_c.items()}); r.update({f"{k}_c": v for k, v in area_c.items()})
            if mask is not None:
                r.update({f"{k}_n": v for k, v in M.localization(n, mask).items()})
            r.update({f"{k}_n": v for k, v in M.activation_area(n).items()})
            r["com_drift_px"] = float(np.linalg.norm(M.center_of_mass(n) - com_c))
            if m == "EigenCAM":
                r["eig_minor_lobe_c"] = X.minor_lobe(signed[0]); r["eig_minor_lobe_n"] = X.minor_lobe(signed[j + 1])
                r["eig_signed_corr"] = M._pearson(signed[0].ravel(), signed[j + 1].ravel())
            main.append(r)
    base = []
    for t, l in conds:
        g = P.rng_for(seed, pid, t, l, "baseline")
        a, b = g.random((224, 224)), g.random((224, 224))
        base.append(dict(Patient=pid, Label=label, Method="Random_Pixel", Type=t, Level=l, **M.xfi(a, b)))
        a7 = cv2.resize(g.random((7, 7)).astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
        b7 = cv2.resize(g.random((7, 7)).astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
        base.append(dict(Patient=pid, Label=label, Method="Random_Smooth", Type=t, Level=l, **M.xfi(a7, b7)))
    u = np.full((224, 224), 0.5)
    base.append(dict(Patient=pid, Label=label, Method="Uniform", Type="none", Level=0, **M.xfi(u, u)))
    for m in RAND_METHODS:
        base.append(dict(Patient=pid, Label=label, Method=f"ModelRandom_{m}", Type="none", Level=0, **M.xfi(maps[m][0], rmaps[m][0])))
    return main, base


# ------------------------------------------------------------------ main
def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=D.ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--save-maps", type=int, default=100, help="patients whose maps are stored (balanced over TP/FN/TN/FP)")
    ap.add_argument("--cam-batch", type=int, default=10, help="images per CAM call (memory bound)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    cfg = D.load_cfg(); seed = cfg["project"]["seed"]; spec = MODELS[a.model]
    out = D.ROOT / (a.out or f"output/phase3/xfi_{a.model}"); out.mkdir(parents=True, exist_ok=True)
    (out / "maps").mkdir(exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} [xfi {a.model}] {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    from threadpoolctl import threadpool_limits
    threadpool_limits(8); torch.set_num_threads(8)        # leave the other cores to the metric workers
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thr = resolve_threshold(spec)
    model = load_model(spec, device); rmodel = X.randomized_copy(model, seed)
    expl, rexpl = X.Explainer(model), X.Explainer(rmodel)
    sp = D.load_splits(cfg, seed=seed); test = sp["test"]
    if a.limit:
        test = test.head(a.limit)
    labels = pd.read_csv(D.path(cfg, "labels_csv"))
    boxes = {pid: g[["x", "y", "width", "height"]].dropna().values.tolist() for pid, g in labels[labels.Target == 1].groupby("patientId")}
    cache = D.ImageCache(cfg)
    ids, y = list(test.patientId), dict(zip(test.patientId, test.Target.astype(int)))

    # clean-image probabilities for all patients (selection of the stored maps; group counts)
    with torch.no_grad():
        pc = {}
        for i in range(0, len(ids), 64):
            b = ids[i:i + 64]
            x = torch.from_numpy(np.stack([to_input(cache.get(p) / 255.0, spec["imagenet_norm"]) for p in b])).to(device)
            pc.update(zip(b, torch.sigmoid(model(x)).squeeze(1).cpu().numpy().tolist()))
    grp = {p: ("TP" if y[p] and pc[p] > thr else "FN" if y[p] else "FP" if pc[p] > thr else "TN") for p in ids}
    g_rng = P.rng_for(seed, "save_maps", a.model)
    save = set()
    for gname in ("TP", "FN", "TN", "FP"):
        members = sorted(p for p in ids if grp[p] == gname)
        save |= set(g_rng.choice(members, size=min(len(members), a.save_maps // 4), replace=False).tolist()) if members else set()
    manifest = dict(model=a.model, spec=spec, threshold=thr, weights_sha256=D.sha256(D.ROOT / spec["weights"]), git=git_rev(),
                    config_sha256=D.sha256(D.ROOT / "configs" / "config.yaml"), n_patients=len(ids),
                    groups={k: sum(1 for v in grp.values() if v == k) for k in ("TP", "FN", "TN", "FP")},
                    saved_maps=sorted(save), seed=seed, torch=torch.__version__, python=platform.python_version(),
                    device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", started=f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"threshold={thr:.4f} n={len(ids)} groups={manifest['groups']} workers={a.workers}")

    conds = P.conditions(); t_start = time.time(); done = 0
    shards = [ids[i:i + a.shard] for i in range(0, len(ids), a.shard)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for si, sids in enumerate(shards):
            f_main, f_base = out / f"results_main_s{si:03d}.csv", out / f"results_base_s{si:03d}.csv"
            if f_main.exists() and f_base.exists():
                log(f"shard {si} exists, skipped"); done += len(sids); continue
            futures, rows_main, rows_base, clean_maps = [], [], [], {}
            for k, pid in enumerate(sids):
                clean = (cache.get(pid) / 255.0).astype(np.float32)
                noisy = np.stack([P.apply(clean, t, l, seed, pid) for t, l in conds])
                x = torch.from_numpy(np.stack([to_input(im, spec["imagenet_norm"]) for im in np.concatenate([clean[None], noisy])])).to(device)
                with torch.no_grad():
                    probs = torch.sigmoid(model(x)).squeeze(1).cpu().numpy().astype(float)
                cls = int(probs[0] > thr)
                maps, signed = expl.explain(x, cls, chunk=a.cam_batch)
                rmaps, _ = rexpl.explain(x[:1], cls, methods=RAND_METHODS, chunk=a.cam_batch)
                clean_maps[pid] = {m: maps[m][0] for m in RAND_METHODS}
                if pid in save:
                    np.savez_compressed(out / "maps" / f"{pid}.npz", clean=(clean * 255).astype(np.uint8), noisy=noisy.astype(np.float16),
                                        probs=probs, cls=cls, boxes=np.array(boxes.get(pid, []), dtype=float).reshape(-1, 4),
                                        **{f"map_{m}": maps[m].astype(np.float16) for m in CAM_METHODS},
                                        **{f"rmap_{m}": rmaps[m].astype(np.float16) for m in RAND_METHODS})
                job = dict(pid=pid, label=y[pid], boxes=boxes.get(pid, []), clean=clean, noisy=noisy, probs=probs, cls=cls, thr=thr,
                           maps=maps, signed=signed, rmaps=rmaps, seed=seed)
                futures.append(pool.submit(compute_rows, job))
                if len(futures) >= 2 * a.workers:
                    mr, br = futures.pop(0).result(); rows_main += mr; rows_base += br
            for f in futures:
                mr, br = f.result(); rows_main += mr; rows_base += br
            # cross-patient baseline within the shard: patient k vs patient k-1 (cyclic)
            for k, pid in enumerate(sids):
                partner = sids[k - 1] if len(sids) > 1 else pid
                for m in RAND_METHODS:
                    rows_base.append(dict(Patient=pid, Label=y[pid], Method=f"CrossPatient_{m}", Type="none", Level=0, partner=partner,
                                          **M.xfi(clean_maps[pid][m], clean_maps[partner][m])))
            pd.DataFrame(rows_main).to_csv(str(f_main) + ".tmp", index=False); os.replace(str(f_main) + ".tmp", f_main)
            pd.DataFrame(rows_base).to_csv(str(f_base) + ".tmp", index=False); os.replace(str(f_base) + ".tmp", f_base)
            done += len(sids); el = time.time() - t_start
            log(f"shard {si + 1}/{len(shards)} written ({done}/{len(ids)} patients, {el / 60:.1f} min, "
                f"ETA {el / max(1, done) * (len(ids) - done) / 60:.1f} min)")

    rm = pd.concat([pd.read_csv(out / f"results_main_s{i:03d}.csv") for i in range(len(shards))], ignore_index=True)
    rb = pd.concat([pd.read_csv(out / f"results_base_s{i:03d}.csv") for i in range(len(shards))], ignore_index=True)
    rm.to_csv(out / "results_main.csv", index=False); rb.to_csv(out / "results_base.csv", index=False)
    manifest.update(finished=f"{datetime.now():%Y-%m-%d %H:%M:%S}", rows_main=len(rm), rows_base=len(rb),
                    results_main_sha256=D.sha256(out / "results_main.csv"), results_base_sha256=D.sha256(out / "results_base.csv"),
                    minutes=round((time.time() - t_start) / 60, 1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"FINISHED rows_main={len(rm)} rows_base={len(rb)} minutes={manifest['minutes']}")


if __name__ == "__main__":
    main()
