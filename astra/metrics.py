"""Feature-space metrics used by Garcia-Jara et al. (2022): density & recall (Eqs. 5-6), the G-score
(Eqs. 13-16) and FID (Eq. 3). Precision and coverage are computed too, for completeness.

A ball B(phi, r) is the sphere around phi whose radius r is the distance to its k-th nearest neighbour
inside the same set (Eq. 4).
"""
import numpy as np
import torch
from scipy import linalg, stats


def knn_radii(f, k, chunk=4096):
    """Distance from every row of f to its k-th nearest neighbour in f (itself excluded)."""
    out = []
    for i in range(0, len(f), chunk):
        d = torch.cdist(f[i:i + chunk], f)
        out.append(d.topk(k + 1, dim=1, largest=False).values[:, k])      # column 0 is the point itself
    return torch.cat(out)


def prdc(real, fake, k=5, real_radii=None, chunk=4096):
    """Precision, recall, density, coverage of fake features against real features (torch, same device)."""
    rr = knn_radii(real, k) if real_radii is None else real_radii
    rf = knn_radii(fake, k)
    in_real_balls = torch.zeros(len(fake), device=real.device)       # how many real balls hold each fake
    real_covered = torch.zeros(len(real), dtype=torch.bool, device=real.device)
    real_recalled = torch.zeros(len(real), dtype=torch.bool, device=real.device)
    for i in range(0, len(real), chunk):
        d = torch.cdist(real[i:i + chunk], fake)                     # (chunk, n_fake)
        inside = d <= rr[i:i + chunk, None]
        in_real_balls += inside.sum(0)
        real_covered[i:i + chunk] = inside.any(1)
        real_recalled[i:i + chunk] = (d <= rf[None, :]).any(1)
    return {"precision": (in_real_balls > 0).float().mean().item(),
            "recall": real_recalled.float().mean().item(),
            "density": (in_real_balls.sum() / (k * len(fake))).item(),
            "coverage": real_covered.float().mean().item()}


def g_score(D, R):
    """D, R: arrays (n_checkpoints, n_classes). Per-class min-max normalisation over the checkpoints of one
    run (Eqs. 13-14), per-class F-score (Eq. 15), macro average (Eq. 16). Returns one score per checkpoint."""
    D, R = np.asarray(D, float), np.asarray(R, float)

    def mm(a):
        lo, hi = a.min(0, keepdims=True), a.max(0, keepdims=True)
        return (a - lo) / np.where(hi > lo, hi - lo, 1.0)

    Dn, Rn = mm(D), mm(R)
    F = np.where(Dn + Rn > 0, 2 * Dn * Rn / np.where(Dn + Rn > 0, Dn + Rn, 1.0), 0.0)
    return F.mean(1)


def gauss_stats(f):
    """Mean and covariance of a feature set (numpy, float64)."""
    return f.mean(0), np.cov(f, rowvar=False)


def fid_from_stats(mu1, s1, mu2, s2):
    """Frechet distance between two Gaussians (Eq. 3)."""
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():                                 # standard fix for a singular product
        eps = np.eye(len(s1)) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1 + s2 - 2 * np.real(covmean)))


def fid(real, fake):
    """Frechet distance between Gaussians fitted to two feature sets (numpy arrays)."""
    return fid_from_stats(*gauss_stats(real), *gauss_stats(fake))


def welch_p(a, b):
    """Two-sided Welch's t-test p-value, as in Table 5."""
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)
