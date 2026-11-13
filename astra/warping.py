"""Soft window-warping baselines of Garcia-Jara et al. (2022), Sec. 4.8.3, Eqs. 17-18.

A random phase window [t1, t2] is picked; values v inside it are mapped by f(v) = a + b*tanh(k*(v - c)),
with a = c = window centre and b = half-width, so the window never grows (f(lo) >= lo, f(hi) <= hi).
k ~ U[1/(2b), 2/b]: small k*b contracts towards the centre, large k*b (saturated tanh) expands.
(The paper prints the interval with `a`; with `a` the magnitude version breaks, because the centre of
median-centred magnitudes can be 0 or negative, so we read it as the half-width b.)
Time-warping maps the phases, magnitude-warping maps the magnitudes (window limits m1, m2 = min/max
magnitude inside the phase window), mixed-warping does both.
"""
import torch


def _tanh_map(v, lo, hi, inside):
    a = (lo + hi) / 2
    b = (hi - lo) / 2
    ok = inside & (b > 0)
    b_safe = torch.where(b > 0, b, torch.ones_like(b))
    k = (0.5 + 1.5 * torch.rand_like(b)) / b_safe          # U[1/(2b), 2/b]
    return torch.where(ok, a + b * torch.tanh(k * (v - a)), v)


def soft_warp(xb, mode):
    """xb: (..., 2, L) with channel 0 = magnitudes, channel 1 = phases. Returns a warped copy."""
    x, ph = xb[..., 0, :], xb[..., 1, :]
    shape = x.shape[:-1] + (1,)
    t = torch.sort(torch.rand(shape[:-1] + (2,), device=xb.device), dim=-1).values
    t1, t2 = t[..., :1], t[..., 1:]
    inside = (ph >= t1) & (ph <= t2)
    if mode in ("time", "mixed"):
        ph = _tanh_map(ph, t1, t2, inside)
    if mode in ("mag", "mixed"):
        big = torch.finfo(x.dtype).max
        m1 = torch.where(inside, x, torch.full_like(x, big)).min(-1, keepdim=True).values
        m2 = torch.where(inside, x, torch.full_like(x, -big)).max(-1, keepdim=True).values
        enough = inside.sum(-1, keepdim=True) >= 2
        m1 = torch.where(enough, m1, torch.zeros_like(m1))
        m2 = torch.where(enough, m2, torch.zeros_like(m2))
        x = _tanh_map(x, m1, m2, inside & enough)
    return torch.stack([x, ph], dim=-2)


def make_warp(mode):
    return None if mode is None else (lambda xb: soft_warp(xb, mode))
