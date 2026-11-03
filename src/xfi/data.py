"""Data access for the v2 pipeline.

- frozen v1 split (output/phase0/split_train.csv, split_test.csv; D-05)
- stratified validation split carved from the v1 training set (D-07), frozen on first use
- 224x224 uint8 image cache built with exactly the v1 preprocessing:
  DICOM pixel_array -> cv2 min-max to 0-255 uint8 -> PIL bilinear resize to 224x224
  (identical to v1: cvtColor GRAY2RGB -> ToPILImage -> Resize((224,224)), applied per channel)
- datasets returning 3-channel tensors, optional ImageNet normalisation
"""
import hashlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[2]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_cfg():
    return yaml.safe_load(open(ROOT / "configs" / "config.yaml"))


def path(cfg, key):
    return ROOT / cfg["paths"][key]


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ splits
def load_splits(cfg, val_frac=0.10, seed=42):
    """Returns dict of DataFrames (patientId, Target): train_full, train_fit, val, test.
    The validation split is created once (stratified on Target) and then always re-read."""
    p0 = path(cfg, "phase0_out")
    train_full = pd.read_csv(p0 / "split_train.csv")
    test = pd.read_csv(p0 / "split_test.csv")
    out = ROOT / "output" / "phase2"; out.mkdir(parents=True, exist_ok=True)
    f_val, f_fit = out / "split_val.csv", out / "split_train_fit.csv"
    if not f_val.exists():
        fit, val = train_test_split(train_full, test_size=val_frac, random_state=seed, stratify=train_full["Target"])
        val.sort_values("patientId").to_csv(f_val, index=False)
        fit.sort_values("patientId").to_csv(f_fit, index=False)
    val, fit = pd.read_csv(f_val), pd.read_csv(f_fit)
    assert set(val.patientId).isdisjoint(fit.patientId)
    assert set(val.patientId) | set(fit.patientId) == set(train_full.patientId)
    assert set(test.patientId).isdisjoint(train_full.patientId)
    return dict(train_full=train_full, train_fit=fit, val=val, test=test)


# ------------------------------------------------------------------ image cache
def preprocess_dicom(fp, size=224):
    """v1 preprocessing up to the resize; returns uint8 (size, size)."""
    img = pydicom.dcmread(fp).pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def _work(args):
    fp, size = args
    return preprocess_dicom(fp, size)


def cache_paths(cfg):
    d = ROOT / "dataset" / "cache224"
    return d, d / "images_u8.npy", d / "ids.csv"


def build_cache(cfg, workers=16, size=224, log=print):
    d, f_img, f_ids = cache_paths(cfg)
    if f_img.exists() and f_ids.exists():
        log(f"[cache] exists: {f_img}")
        return
    d.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(path(cfg, "labels_csv"))
    ids = list(labels["patientId"].unique())
    img_dir = path(cfg, "image_dir")
    arr = np.lib.format.open_memmap(str(f_img) + ".tmp", mode="w+", dtype=np.uint8, shape=(len(ids), size, size))
    with Pool(workers) as pool:
        for i, im in enumerate(pool.imap(_work, [(img_dir / f"{pid}.dcm", size) for pid in ids], chunksize=64)):
            arr[i] = im
            if (i + 1) % 2000 == 0 or i + 1 == len(ids):
                log(f"[cache] {i + 1}/{len(ids)}")
    arr.flush(); del arr
    Path(str(f_img) + ".tmp").rename(f_img)
    pd.DataFrame({"patientId": ids}).to_csv(f_ids, index=False)
    log(f"[cache] written {f_img} sha256={sha256(f_img)}")


class ImageCache:
    def __init__(self, cfg):
        _, f_img, f_ids = cache_paths(cfg)
        self.arr = np.load(f_img, mmap_mode="r")
        self.index = {pid: i for i, pid in enumerate(pd.read_csv(f_ids)["patientId"])}

    def get(self, pid):
        return np.array(self.arr[self.index[pid]])


# ------------------------------------------------------------------ datasets
def make_transform(train, imagenet_norm):
    t = [T.ToPILImage()]
    if train:
        t += [T.RandomHorizontalFlip(), T.RandomRotation(10)]
    t += [T.ToTensor()]
    if imagenet_norm:
        t += [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(t)


class CXRDataset(Dataset):
    """Returns (3x224x224 float tensor, label float, patientId). Images come from the cache."""
    def __init__(self, df, cache, train, imagenet_norm):
        self.ids = list(df["patientId"]); self.y = df["Target"].astype(np.float32).values
        self.cache = cache; self.tf = make_transform(train, imagenet_norm)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        g = self.cache.get(self.ids[i])
        rgb = np.repeat(g[:, :, None], 3, axis=2)
        return self.tf(rgb), torch.tensor(self.y[i]), self.ids[i]
