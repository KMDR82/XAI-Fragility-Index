"""Unit tests of the v2 core (metrics, perturbations, XAI helpers)."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision.models as models

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfi import metrics as M, perturb as P, xai as X
from xfi.run_xfi import to_input

rng = np.random.default_rng(0)


# ------------------------------------------------------------------ XFI
def smooth(seed):
    import cv2
    g = np.random.default_rng(seed).random((7, 7)).astype(np.float32)
    return cv2.resize(g, (224, 224))


def test_xfi_identical_is_zero():
    h = smooth(1)
    assert M.xfi(h, h)["xfi"] == pytest.approx(0.0, abs=1e-9)


def test_xfi_uncorrelated_is_two_thirds():
    v = [M.xfi(rng.random((224, 224)), rng.random((224, 224)))["xfi"] for _ in range(5)]
    assert np.mean(v) == pytest.approx(2 / 3, abs=0.01)


def test_xfi_exceeds_one_for_inverted_map_and_is_bounded():
    h = smooth(2)
    r = M.xfi(h, 1 - h)
    assert r["p_corr"] == pytest.approx(-1, abs=1e-6) and r["xfi"] > 1.0 and r["xfi"] <= 4 / 3 + 1e-9


def test_xfi_constant_map_convention():
    u = np.full((224, 224), 0.5)
    r = M.xfi(u, u)
    assert r["ssim"] == 0.0 and np.isnan(r["p_corr"]) and r["xfi"] == pytest.approx(2 / 3) and r["degenerate"] == 1
    assert M.xfi(smooth(1), smooth(2))["degenerate"] == 0


def test_xfi_matches_v1_definition():
    from scipy.stats import pearsonr, spearmanr
    from skimage.metrics import structural_similarity
    a, b = smooth(3), smooth(4)
    n = lambda h: (h - h.min()) / (h.max() - h.min() + 1e-8)
    c, d = n(a), n(b)
    rs = (structural_similarity(c, d, data_range=1.0) + (pearsonr(c.ravel(), d.ravel())[0] + 1) / 2
          + (spearmanr(c.ravel(), d.ravel())[0] + 1) / 2) / 3
    assert M.xfi(a, b)["xfi"] == pytest.approx(1 - rs, abs=1e-6)


# ------------------------------------------------------------------ localization
def test_gt_mask_scaling():
    m = M.gt_mask([[512, 256, 256, 512]])          # x, y, w, h in 1024 px
    ys, xs = np.where(m)
    assert (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1) == (112, 168, 56, 168)


def test_localization_perfect_map():
    m = M.gt_mask([[256, 256, 256, 256]])
    h = m.astype(float)
    loc = M.localization(h, m)
    assert loc["iou_abs5"] == 1.0 and loc["far_bg_abs5"] == 0.0 and loc["energy_out_box"] == pytest.approx(0.0)
    assert loc["prec_in_box_abs7"] == 1.0


def test_iou_is_against_mask_not_between_maps():
    m = M.gt_mask([[0, 0, 512, 512]])
    h1 = m.astype(float); h2 = (~m).astype(float)
    assert M.localization(h1, m)["iou_abs5"] == 1.0 and M.localization(h2, m)["iou_abs5"] == 0.0


def test_far_bg_absolute():
    m = M.gt_mask([[0, 0, 512, 1024]])             # left half is the box
    h = np.zeros((224, 224)); h[:, 112:168] = 1.0  # half of the background activated
    assert M.localization(h, m)["far_bg_abs5"] == pytest.approx(0.5)


# ------------------------------------------------------------------ perturbations
def test_perturbations_deterministic_and_keyed():
    img = rng.random((224, 224)).astype(np.float32)
    for t, l in P.conditions():
        a, b = P.apply(img, t, l, 42, "p1"), P.apply(img, t, l, 42, "p1")
        assert np.array_equal(a, b) and a.min() >= 0 and a.max() <= 1
    assert not np.array_equal(P.apply(img, "Gaussian", 2, 42, "p1"), P.apply(img, "Gaussian", 2, 42, "p2"))


def test_gaussian_sigma_and_saltpepper_amount():
    img = np.full((224, 224), 0.5, np.float32)
    g = P.apply(img, "Gaussian", 1, 42, "x")
    assert np.std(g - img) == pytest.approx(0.1, rel=0.03)
    img2 = rng.random((224, 224)).astype(np.float32)
    s = P.apply(img2, "SaltPepper", 3, 42, "x")
    assert 0.02 < (s != img2).mean() <= 0.03


def test_input_is_same_in_three_channels():
    g = P.apply(rng.random((224, 224)).astype(np.float32), "Gaussian", 3, 42, "x")
    x = to_input(g, False)
    assert np.array_equal(x[0], x[1]) and np.array_equal(x[1], x[2])
    xn = to_input(g, True)
    assert not np.allclose(xn[0], xn[1])        # normalisation differs per channel, content does not
    assert np.allclose(xn[0] * 0.229 + 0.485, xn[1] * 0.224 + 0.456, atol=1e-6)


def test_input_quality():
    a = rng.random((224, 224)).astype(np.float32)
    psnr, ssim = P.input_quality(a, a)
    assert psnr == float("inf") and ssim == pytest.approx(1.0)


# ------------------------------------------------------------------ XAI helpers
def test_signed_projection_matches_library_full_svd():
    for k in range(5):
        acts = np.maximum(rng.normal(size=(3, 2048, 7, 7)), 0).astype(np.float32)
        assert np.allclose(X.signed_projection(acts), X.LIBRARY_GET_2D_PROJECTION(acts), atol=1e-4)
        assert np.allclose(X.signed_projection_sign_corrected(acts), X.LIBRARY_GET_2D_PROJECTION_SC(acts), atol=1e-4)


def test_eigencam_uses_fast_projection():
    import pytorch_grad_cam.eigen_cam as ec
    assert ec.get_2d_projection is X.signed_projection


def _tiny_resnet():
    m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, 1); return m.eval()


def test_randomized_copy_changes_only_top():
    m = _tiny_resnet(); r1, r2 = X.randomized_copy(m, 1), X.randomized_copy(m, 1)
    assert torch.equal(m.layer1[0].conv1.weight, r1.layer1[0].conv1.weight)
    assert not torch.equal(m.layer4[0].conv1.weight, r1.layer4[0].conv1.weight)
    assert not torch.equal(m.fc.weight, r1.fc.weight)
    assert torch.equal(r1.layer4[2].conv3.weight, r2.layer4[2].conv3.weight)    # deterministic


def test_explainer_shapes():
    m = _tiny_resnet(); e = X.Explainer(m)
    x = torch.from_numpy(np.stack([to_input(rng.random((224, 224)).astype(np.float32), False) for _ in range(3)]))
    maps, signed = e.explain(x, 1)
    assert set(maps) == set(X.Explainer.METHODS) and all(v.shape == (3, 224, 224) for v in maps.values())
    assert signed.shape == (3, 7, 7)
    assert all(v.min() >= 0 and v.max() <= 1 + 1e-6 for v in maps.values())


def test_minor_lobe():
    p = np.array([[3.0, -1.0], [-1.0, -1.0]])
    assert X.minor_lobe(p) == 0 and X.minor_lobe(-p) == 1
