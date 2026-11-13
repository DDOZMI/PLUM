"""Leveling (paper Sec. 4.3): strata, the per-stratum line in log area, and the rank against the reference detections.

  stratum g(b) = 2 * k(b) + t(b)   k = FPN level 0..3 (P2..P5, Eq. level), t = border contact (box within 1 px of an image side)
  line       : per stratum OLS of the raw score on [1, log area] over D_L                     (ols_bins, Eq. ols)
  score      : residual S = s - theta_g - eta_g log a                                         (ols_bins, Eq. resid)
  output     : u = (1 + #{b' in D_R^g : S_b' <= S}) / (|D_R^g| + 1)   (qlvl, Eq. plum; searchsorted side='right')
Boxes are in network-input coordinates; hw = resized (H, W) of the row's image (network input size).
"""
import numpy as np
import torch


def log_area(b):
    return torch.log(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).clamp(min=1e-6)).double().numpy()


def touch(b, hw):
    """1.0 if the box lies within 1 px of an image side (distance to the nearest side < 1, clamped at 0), else 0.0."""
    b = b.double()
    H, W = hw[:, 0].double(), hw[:, 1].double()
    m = torch.stack([b[:, 0], b[:, 1], W - b[:, 2], H - b[:, 3]], 1).min(1).values.clamp(min=0)
    return (m < 1).double().numpy()


def covariates(box, hw):
    return {"la": log_area(box), "touch": touch(box, hw)}


def pooler_level(b):
    """detectron2 assign_boxes_to_levels (min 2, max 5, canonical 224 at level 4), returned as 0..3 (used for DDETR boxes;
    the Faster R-CNN level is the box pooler's own assignment, stored in the dump)."""
    s = np.sqrt(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).double().numpy())
    lv = np.floor(4 + np.log2(s / 224 + 1e-8))
    return (np.clip(lv, 2, 5) - 2).astype(int)


def ols_bins(s_ref, X_ref, l_ref, s, X, l):
    """per-bin OLS of s on [1, X] over reference rows; returns (ref residuals, test residuals)."""
    rr, rt = np.empty(len(s_ref)), np.empty(len(s))
    for b in np.unique(l_ref):
        m, mt = l_ref == b, l == b
        A = np.c_[np.ones(m.sum()), X_ref[m]]
        beta = np.linalg.lstsq(A, s_ref[m], rcond=None)[0]
        rr[m] = s_ref[m] - A @ beta
        rt[mt] = s[mt] - np.c_[np.ones(mt.sum()), X[mt]] @ beta
    assert set(np.unique(l)) <= set(np.unique(l_ref))
    return rr, rt


def qlvl(ref, lref, x, lx):
    u = np.empty(len(x))
    for lv in np.unique(lx):
        r = np.sort(ref[lref == lv]); m = lx == lv
        assert len(r) > 0, f"no reference rows at level {lv}"
        u[m] = (np.searchsorted(r, x[m], side="right") + 1) / (len(r) + 1)
    return u


def level(s_ref, la_ref, lab_ref, fit, rank, s_k, la_k, lab_k):
    """s_ref / la_ref / lab_ref: raw score, log area, stratum of the odd-image reference rows; fit / rank: masks of D_L / D_R
    over those rows; s_k / la_k / lab_k: the same for the test rows. Returns u of the test rows."""
    _, rB = ols_bins(s_ref[fit], la_ref[fit][:, None], lab_ref[fit], s_ref[rank], la_ref[rank][:, None], lab_ref[rank])
    return qlvl(rB, lab_ref[rank], ols_bins(s_ref[fit], la_ref[fit][:, None], lab_ref[fit], s_k, la_k[:, None], lab_k)[1], lab_k)
