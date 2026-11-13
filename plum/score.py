"""Raw PLUM score (paper Sec. 4.1-4.2): unit-norm feature, class means, one shared Ledoit-Wolf covariance, negative Mahalanobis
distance to the nearest class mean.

  z = phi / ||phi||                                  (l2, Eq. feature)
  mu_c = mean of z over the even-image detections predicted as c          (class_means, Eq. mean; float64 accumulation)
  Sigma_0 = scatter of z around the class means                          (scatter, Eq. scatter; float64)
  Sigma = (1 - rho) Sigma_0 + rho m I                                    (ledoit_wolf, Eq. lw / lw-weight)
  s = -min_c (z - mu_c)^T Sigma^-1 (z - mu_c)                            (maha_scores, Eq. dist; float64 quadratic forms,
                                                                           returned as float32)
"""
import torch

DEV = "cuda"


def l2(x):
    x = x.float().to(DEV)
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)


def scatter(X, cls=None, mu=None, idx=None, chunk=65536):
    """S = Z^T Z / n and sum_k ||z_k||^4 with z_k = x_k - mu[cls_k] (or x_k if mu is None), over rows idx (all if
    None). Accumulated in float64 chunk by chunk without materialising the centred copy."""
    p = X.shape[1]
    idx = torch.arange(len(X), device=X.device) if idx is None else idx
    S = torch.zeros(p, p, dtype=torch.float64, device=X.device)
    q4 = torch.zeros((), dtype=torch.float64, device=X.device)
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        z = X[j].double()
        if mu is not None:
            z -= mu[cls[j]].double()
        S += z.T @ z
        q4 += (z.pow(2).sum(1) ** 2).sum()
    return S / len(idx), q4


def ledoit_wolf(S, q4, n):
    """Ledoit & Wolf (2004) shrinkage towards mu*I (closed form, no hyperparameter) from scatter S and sum ||x||^4."""
    p = S.shape[0]
    mu = torch.trace(S) / p
    d2 = ((S - mu * torch.eye(p, device=S.device, dtype=S.dtype)) ** 2).sum()
    b2bar = (q4 - n * (S ** 2).sum()) / n ** 2
    b2 = torch.minimum(b2bar, d2)
    a = (b2 / d2).item()
    return a * mu * torch.eye(p, device=S.device, dtype=S.dtype) + (1 - a) * S, a


def precision(X, kind, cls=None, mu=None, idx=None):
    S, q4 = scatter(X, cls, mu, idx)
    n = len(X) if idx is None else len(idx)
    if kind == "lw":
        Sl, a = ledoit_wolf(S, q4, n)
        return torch.linalg.inv(Sl), {"lw_shrinkage": a}
    return torch.linalg.pinv(S, hermitian=True), {}  # not used by PLUM (kind is always 'lw')


def class_means(x, c, K):
    mu = torch.zeros(K, x.shape[1], device=x.device, dtype=torch.float64)
    n = torch.bincount(c, minlength=K)
    mu.index_add_(0, c, x.double())  # float64 accumulation
    return (mu / n.clamp(min=1)[:, None]).float(), n


def fit_pooled(tr, K, cov):
    x, c = tr["xf"], tr["cls"].to(DEV)
    mu, n = class_means(x, c, K)
    P, info = precision(x, cov, c, mu)
    return {"mu": mu, "valid": n > 0, "P": P, "info": info}


def maha_scores(x, mu, P, valid, chunk=65536):
    """float64 quadratic forms; returns -min over valid classes (higher = more ID), as float32."""
    out = []
    for i in range(0, len(x), chunk):
        xi = x[i:i + chunk].double()
        d = []
        for k in range(mu.shape[0]):
            if not valid[k]:
                continue
            z = xi - mu[k].double()
            d.append(((z @ P) * z).sum(1))
        out.append(-torch.stack(d, 1).min(1).values)
    return torch.cat(out).float()
