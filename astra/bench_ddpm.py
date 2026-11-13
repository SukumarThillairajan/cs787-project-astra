"""Benchmark family 3 — diffusion: a conditional DDPM (Ho et al. 2020) with a DDIM sampler (Song et al. 2021).

Forward process: a real curve x0 is mixed with Gaussian noise in T = 1000 steps,
    x_t = sqrt(ab_t) * x0 + sqrt(1 - ab_t) * eps,    eps ~ N(0, I),
where ab_t falls from 1 (clean) to about 0 (pure noise) along the cosine schedule of Nichol & Dhariwal (2021).
Training: pick t at random, and train the network to predict the noise eps from x_t (mean squared error).
Sampling: start from pure noise and remove it step by step. DDIM does this deterministically and can skip
steps, so a curve costs `nfe` network calls (default 32) instead of 1000.
Each step first estimates the clean curve, x0 = (x_t - sqrt(1 - ab_t) * eps) / sqrt(ab_t). At the very last
steps of the cosine schedule sqrt(ab_t) is about 5e-5, so this divides by almost zero and the estimate is
rounding noise. Sampling therefore starts at the noisiest step with ab_t >= 0.001 (step 979 of 1000 for the
cosine schedule), where x_t is still 99.9% noise. Training uses all 1000 steps.

The network (astra.bench_net.UNet1d) and its conditioning are shared with the flow-matching model.
In 10% of the training samples the class is hidden, which allows classifier-free guidance at sampling time.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bench_net import UNet1d, guided
from . import bench_grid
from .bench_run import Family, grid_jobs
from .classifier import autocast

DEFAULTS = dict(batch=256, lr=2e-4, base=88, T=1000, schedule="cosine", pred="eps", p_uncond=0.1, grad_clip=1.0,
                nfe=32, guidance=1.0, chunk=20, use_phase=True,
                ab_min=1e-3)       # ours: sampling starts at the noisiest step that still has ab_t >= ab_min
GRID, FINAL, SPEED = bench_grid.GRID["ddpm"], bench_grid.FINAL["ddpm"], bench_grid.SPEED["ddpm"]   # astra/bench_grid.py
JOBS = grid_jobs("ddpm", GRID, FINAL)
TINY = dict(nfe=4)
LOSS_T = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)             # noise levels (t / T) at which losses() is measured


def alpha_bar(T, schedule):
    """ab_t for t = 0 .. T-1: the share of the signal's variance that is left after t+1 noising steps."""
    if schedule == "cosine":
        t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
        f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    elif schedule == "linear":
        betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
    else:
        raise ValueError(schedule)
    return torch.cumprod(1 - betas, 0).float()


class MODEL(Family):
    SWEEPS = ({"nfe": 8}, {"guidance": 2.0}, {"nfe": 4}, {"nfe": 16}, {"nfe": 64}, {"guidance": 1.5})

    def __init__(self, cfg, data, device):
        super().__init__(cfg, data, device)
        self.T = cfg["T"]
        self.ab = alpha_bar(self.T, cfg["schedule"]).to(device)
        self.t_start = int(torch.nonzero(self.ab >= cfg["ab_min"]).max())      # first step of every sampling run
        self.net = UNet1d(data["n_classes"], cfg["base"], use_phase=cfg["use_phase"]).to(device)
        self.ema = self.make_ema(self.net)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.track(net=self.net, ema=self.ema, opt=self.opt, scaler=self.scaler)

    def eval_nets(self):
        return {"ema": self.ema}

    def noised(self, x0, t, eps):
        a, s = self.ab[t].sqrt()[:, None], (1 - self.ab[t]).sqrt()[:, None]
        return a * x0 + s * eps, (eps if self.cfg["pred"] == "eps" else a * eps - s * x0)

    def step(self, draw, it):
        c = self.cfg
        idx = draw()
        x0, (ph, camp, y) = self.data["u"][idx], self.cond(idx)
        t = torch.randint(0, self.T, (len(idx),), device=self.dev)
        xt, target = self.noised(x0, t, torch.randn_like(x0))
        y = torch.where(torch.rand(len(idx), device=self.dev) < c["p_uncond"], torch.full_like(y, self.net.n_classes), y)
        with autocast(self.dev, self.amp):
            out = self.net(xt, t.float() / self.T, ph, camp, y)
        loss = F.mse_loss(out.float(), target)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(self.net.parameters(), c["grad_clip"])
        self.scaler.step(self.opt)
        self.scaler.update()
        self.update_ema()
        return torch.nan_to_num(loss.detach())[None]

    def sample_batch(self, j, g, nfe=None, guidance=None):
        """DDIM with eta = 0: nfe evenly spaced time steps from t_start down to 0."""
        c = self.cfg
        nfe, w = nfe or c["nfe"], c["guidance"] if guidance is None else guidance
        ph, camp, y = self.cond(j)
        x = torch.randn(len(j), self.data["L"], device=self.dev, generator=g)
        ts = torch.linspace(self.t_start, 0, nfe).round().long().tolist()
        clip = self.data["out_scale"]
        for i, t in enumerate(ts):
            ab_t = self.ab[t]
            ab_prev = self.ab[ts[i + 1]] if i + 1 < nfe else torch.ones((), device=self.dev)
            tt = torch.full((len(j),), t / self.T, device=self.dev)
            with autocast(self.dev, self.amp):
                out = guided(self.ema, x, tt, ph, camp, y, w)
            out = out.float()
            x0 = (x - (1 - ab_t).sqrt() * out) / ab_t.sqrt() if c["pred"] == "eps" else ab_t.sqrt() * x - (1 - ab_t).sqrt() * out
            x0 = x0.clamp(-clip, clip)                     # real curves lie in [-2, 2]
            eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return self.to_mag(x, j)

    @torch.no_grad()
    def losses(self, idx, seed):
        """The training loss of the EMA network at six fixed noise levels, with fixed noise."""
        g = torch.Generator(device=self.dev).manual_seed(seed)
        x0, (ph, camp, y) = self.data["u"][idx], self.cond(idx)
        tot = 0.0
        for frac in LOSS_T:
            t = torch.full((len(idx),), int(frac * self.T), device=self.dev)
            xt, target = self.noised(x0, t, torch.randn(x0.shape, device=self.dev, generator=g))
            tot += float(F.mse_loss(self.ema(xt, t.float() / self.T, ph, camp, y), target))
        return tot / len(LOSS_T)
