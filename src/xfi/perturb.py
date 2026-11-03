"""Image perturbations of the stress test (D-10).

Inputs and outputs are single-channel float32 images in [0, 1] (224 x 224). The caller replicates
the result to three channels, so every channel carries the same corruption (fix of A-08: v1 drew
independent Gaussian noise per RGB channel).

Parameters are those of v1:  Gaussian sigma = 0.1 * L,  blur kernel k = 2L + 1,  salt and pepper amount = 0.01 * L
(salt value = 99th percentile, pepper value = 1st percentile of the image; num//2 positions each, drawn with replacement).

Randomness: every (patient, type, level) gets its own generator derived from the project seed, so results do not
depend on processing order and an interrupted run resumes bit-identically (v1 used one global stream).
"""
import hashlib
import cv2
import numpy as np

TYPES = ("Gaussian", "Blur", "SaltPepper")
LEVELS = (1, 2, 3)


def rng_for(seed, *keys):
    h = hashlib.sha256("|".join(map(str, (seed,) + keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def gaussian(img, level, rng):
    return np.clip(img + rng.normal(0.0, 0.1 * level, img.shape), 0, 1).astype(np.float32)


def blur(img, level, rng=None):
    k = int(2 * level + 1)
    return cv2.GaussianBlur(img.astype(np.float32), (k, k), 0)


def salt_pepper(img, level, rng):
    res = img.astype(np.float32).copy()
    h, w = img.shape
    salt, pepper = np.percentile(img, 99), np.percentile(img, 1)
    n = int(0.01 * level * h * w)
    res[rng.integers(0, h, n // 2), rng.integers(0, w, n // 2)] = salt
    res[rng.integers(0, h, n // 2), rng.integers(0, w, n // 2)] = pepper
    return res


FUNCS = {"Gaussian": gaussian, "Blur": blur, "SaltPepper": salt_pepper}


def apply(img, ptype, level, seed, pid):
    return FUNCS[ptype](img, level, rng_for(seed, pid, ptype, level, "perturb"))


def conditions():
    return [(t, l) for t in TYPES for l in LEVELS]


def input_quality(clean, noisy):
    """PSNR (dB) and SSIM between clean and perturbed input images (D-17)."""
    from skimage.metrics import structural_similarity
    mse = float(np.mean((clean - noisy) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
    return psnr, float(structural_similarity(clean, noisy, data_range=1.0))
