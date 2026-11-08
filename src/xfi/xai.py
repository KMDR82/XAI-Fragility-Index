"""Explanation methods (grad-cam 1.5.7) with batch evaluation and Eigen-CAM sign diagnostics (D-11, A-10).

GradCAM     : gradient-weighted, explains the given target class.
EigenCAM    : first principal component of the activations; library default (no sign correction); class-agnostic.
EigenCAM_SC : same, with the library's sign correction (flip when |min| > |max|); ablation only.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
import pytorch_grad_cam.eigen_cam as _eigen_cam_module
from pytorch_grad_cam.utils import svd_on_activations as _svd

# grad-cam 1.5.7 computes the full SVD (full_matrices=True) of the (h*w) x C activation matrix, i.e. also the unused
# C x C right factor (2048 x 2048 for ResNet-50): ~0.36 s per image. Only the first right singular vector is used, and
# the economy SVD returns the same vector, sign included (verified on 330 activation sets and in tests/test_core.py),
# ~50x faster. The library function is replaced by the economy version; the original is kept for the tests.
LIBRARY_GET_2D_PROJECTION = _svd.get_2d_projection
LIBRARY_GET_2D_PROJECTION_SC = _svd.get_2d_projection_with_sign_correction


def signed_projection(acts):
    """Signed first-PC projection as grad-cam's get_2d_projection (centring, first right singular vector), before ReLU."""
    acts = np.array(acts, dtype=np.float32, copy=True); acts[np.isnan(acts)] = 0
    out = []
    for a in acts:
        r = a.reshape(a.shape[0], -1).T
        r = r - r.mean(axis=0)
        _, _, vt = np.linalg.svd(r, full_matrices=False)
        out.append((r @ vt[0]).reshape(a.shape[1:]))
    return np.float32(out)


def signed_projection_sign_corrected(acts):
    p = signed_projection(acts)
    flip = np.abs(p.reshape(len(p), -1).min(1)) > np.abs(p.reshape(len(p), -1).max(1))
    p[flip] = -p[flip]
    return p


_eigen_cam_module.get_2d_projection = signed_projection


class EigenCAMSC(EigenCAM):
    def get_cam_image(self, input_tensor, target_layer, target_category, activations, grads, eigen_smooth):
        return signed_projection_sign_corrected(activations)


class Explainer:
    METHODS = ("GradCAM", "EigenCAM", "EigenCAM_SC")

    def __init__(self, model):
        layers = [model.layer4[-1]]
        self.cams = {"GradCAM": GradCAM(model=model, target_layers=layers),
                     "EigenCAM": EigenCAM(model=model, target_layers=layers),
                     "EigenCAM_SC": EigenCAMSC(model=model, target_layers=layers)}

    def explain(self, x, target_class, methods=METHODS, chunk=10):
        """x: (N, 3, 224, 224) on device. Returns {method: (N, 224, 224) float32} and Eigen-CAM signed projections (N, 7, 7).
        Grad-CAM runs with gradients; the Eigen-CAM variants run without (they do not use gradients). `chunk` bounds memory."""
        maps, signed = {m: [] for m in methods}, []
        for i in range(0, x.shape[0], chunk):
            xb = x[i:i + chunk]
            targets = [BinaryClassifierOutputTarget(int(target_class))] * xb.shape[0]
            for m in methods:
                with torch.set_grad_enabled(m == "GradCAM"):
                    maps[m].append(self.cams[m](input_tensor=xb, targets=targets).astype(np.float32))
                if m == "EigenCAM":
                    signed.append(signed_projection(self.cams[m].activations_and_grads.activations[0].numpy()))
        return {m: np.concatenate(v) for m, v in maps.items()}, (np.concatenate(signed) if signed else None)


def minor_lobe(p):
    """1 if |min| > |max| of the signed projection: the ReLU in grad-cam keeps the minor lobe (the case the sign
    correction flips). The projection has zero mean by construction, so shares of positive mass are uninformative."""
    return int(abs(float(p.min())) > abs(float(p.max())))


def randomized_copy(model, seed):
    """Cascading randomization of the top of the network (fc, then all of layer4), Adebayo et al. sanity check."""
    m = copy.deepcopy(model)
    g = torch.random.fork_rng(devices=[] if not next(model.parameters()).is_cuda else [next(model.parameters()).device])
    with g:
        torch.manual_seed(seed)
        for mod in list(m.layer4.modules()) + [m.fc]:
            if isinstance(mod, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                mod.reset_parameters()
                if isinstance(mod, nn.BatchNorm2d):
                    mod.reset_running_stats()
    return m.eval()
