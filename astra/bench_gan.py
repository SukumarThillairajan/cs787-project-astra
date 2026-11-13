"""Benchmark family 1 — GAN: the conditional WGAN-GP of Garcia-Jara et al. (2022).

Networks, losses and optimiser settings are those of the replication notebook (astra.gan): generator
G(z | phases, amplitude, class), critic with a real-vs-fake score and a class output (AC-GAN), 5 critic steps per
generator step, gradient penalty, Adam(1e-4, 0.5, 0.9), batch 64, mixed precision.
What differs from the replication is common to all four families: the split over stars, the target (the
amplitude-normalised curve), the training schedule, and the checkpoint selection.
"""
import torch
import torch.nn.functional as F

from .bench_data import LATENT
from . import bench_grid
from .bench_run import Family, grid_jobs
from .classifier import autocast
from .gan import Discriminator, Generator, gradient_penalty

DEFAULTS = dict(latent=None, batch=64, lr=1e-4, n_disc=5, lambda_gp=10.0, xi=1e-3, eps_drift=1.0, chunk=10)
GRID, FINAL, SPEED = bench_grid.GRID["gan"], bench_grid.FINAL["gan"], bench_grid.SPEED["gan"]   # astra/bench_grid.py
JOBS = grid_jobs("gan", GRID, FINAL)
TINY = {}


class MODEL(Family):
    STATS = ("loss_D", "loss_G", "W_est")
    SWEEPS = ({"temp": 0.7},)          # noise drawn from a narrower Gaussian (the "truncation trick")

    def __init__(self, cfg, data, device):
        super().__init__(cfg, data, device)
        N, L = data["n_classes"], data["L"]
        self.latent = cfg.get("latent") or LATENT.get(L, 16)
        self.G = Generator(self.latent, L, N, data["out_scale"]).to(device)
        self.D = Discriminator(L, N).to(device)
        self.G_ema = self.make_ema(self.G)
        self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg["lr"], betas=(0.5, 0.9))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg["lr"], betas=(0.5, 0.9))
        self.scalerG = torch.amp.GradScaler(enabled=self.amp)
        self.scalerD = torch.amp.GradScaler(enabled=self.amp)
        self.track(G=self.G, D=self.D, G_ema=self.G_ema, optG=self.optG, optD=self.optD, scalerG=self.scalerG,
                   scalerD=self.scalerD)

    def eval_nets(self):
        return {"G_ema": self.G_ema, "D": self.D}

    def cond(self, idx):                # these networks take the class as a one-hot vector
        d = self.data
        return d["ph"][idx], d["camp"][idx], d["onehot"][idx]

    def step(self, draw, it):
        c, d = self.cfg, self.data
        for _ in range(c["n_disc"]):
            idx = draw()
            x_r, cond, y = d["u"][idx], self.cond(idx), d["y"][idx]
            with torch.no_grad(), autocast(self.dev, self.amp):
                x_g = self.G(torch.randn(len(idx), self.latent, device=self.dev), *cond).float()
            with autocast(self.dev, self.amp):
                d_r, logit_r = self.D(x_r, *cond)
                d_g, logit_g = self.D(x_g, *cond)
            d_r, d_g = d_r.float(), d_g.float()
            gp = gradient_penalty(self.D, x_r, x_g, cond, self.scalerD, self.amp)
            h = F.cross_entropy(logit_r.float(), y) + F.cross_entropy(logit_g.float(), y)
            loss_d = d_g.mean() - d_r.mean() + c["lambda_gp"] * gp + c["xi"] * h + c["eps_drift"] * (d_r ** 2).mean()
            self.optD.zero_grad(set_to_none=True)
            self.scalerD.scale(loss_d).backward()
            self.scalerD.step(self.optD)
            self.scalerD.update()
        idx = draw()
        cond, y = self.cond(idx), d["y"][idx]
        with autocast(self.dev, self.amp):
            x_g = self.G(torch.randn(len(idx), self.latent, device=self.dev), *cond)
            d_g, logit_g = self.D(x_g, *cond)
        d_g = d_g.float()
        loss_g = -d_g.mean() + c["xi"] * F.cross_entropy(logit_g.float(), y)
        self.optG.zero_grad(set_to_none=True)
        self.scalerG.scale(loss_g).backward()
        self.scalerG.step(self.optG)
        self.scalerG.update()
        self.update_ema()
        return torch.nan_to_num(torch.stack([loss_d.detach(), loss_g.detach(), (d_r.mean() - d_g.mean()).detach()]))

    def sample_batch(self, j, g, temp=1.0):
        z = temp * torch.randn(len(j), self.latent, device=self.dev, generator=g)
        return self.to_mag(self.G_ema(z, *self.cond(j)), j)

    @torch.no_grad()
    def losses(self, idx, seed):
        """Minus the critic's mean score of real curves. If it is lower on training stars than on validation
        stars, the critic has learnt to recognise its training set (over-fitting)."""
        return float(-self.D(self.data["u"][idx], *self.cond(idx))[0].mean())
