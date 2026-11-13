"""PLUM end to end on loaded dumps (paper Sec. 4): split, statistics, raw score, strata, lines, ranks.

Split of the ID training reference images by image index i (position in data.train_subset):
  i even          -> class means and shared Ledoit-Wolf covariance (D_S)
  i odd, i%4 == 1 -> per-stratum lines in log area (D_L)
  i odd, i%4 == 3 -> reference residuals for the rank (D_R)
Strata = 2 * FPN level + border contact (8 strata, no merging). Test rows: every ID-val / OOD detection of the protocol;
DDETR test scores are computed per unique query and expanded to the pair rows (r2u).
"""
import numpy as np
import torch

from plum.level import level
from plum.score import DEV, fit_pooled, l2, maha_scores


def expand(v, r):
    return v if r is None else v[r]


@torch.no_grad()
def plum_values(X):
    """X = plum.load.load_frcnn / load_ddetr output. Returns (u per set, stratum per set, raw score per set)."""
    K, sets, cr, ct, L, r2u = (X[k] for k in ("K", "sets", "cr", "ct", "L", "r2u"))
    ref_x, te_x = X["ref_x"], X["te_x"]
    si = X["split_idx"]
    ev, odd = si % 2 == 0, si % 2 == 1
    io = si[odd]
    fit, rank = io % 4 == 1, io % 4 == 3
    evt, oddt = torch.from_numpy(ev), torch.from_numpy(odd)
    st = fit_pooled({"xf": l2(ref_x[evt]), "cls": X["cls_ref"][evt].to(DEV)}, K, "lw")
    score = lambda x: maha_scores(l2(x), st["mu"], st["P"], st["valid"]).cpu().numpy()  # noqa: E731
    s_ref = score(ref_x[oddt]).astype(np.float64)
    s_te = {k: expand(score(te_x[k]).astype(np.float64), r2u[k]) for k in sets}
    lvl_ref = X["lvl_ref"][odd]
    lab_ref = lvl_ref * 2 + cr["touch"][odd].astype(int)
    la_ref = cr["la"][odd]
    C = {k: L[k] * 2 + ct[k]["touch"].astype(int) for k in sets}
    U = {k: level(s_ref, la_ref, lab_ref, fit, rank, s_te[k], ct[k]["la"], C[k]) for k in sets}
    return U, C, s_te
