"""Benchmark family 4 — flow matching (Lipman et al. 2023; the straight-line "rectified flow" path of
Liu et al. 2023).

Idea: move noise to data along straight lines. Draw noise x0 ~ N(0, I) and a real curve x1, and put a point
on the line between them at a random time t in [0, 1]:
    x_t = (1 - t) * x0 + t * x1.
Along that line the velocity is constant, x1 - x0. Training: the network sees (x_t, t, condition) and is
trained to predict this velocity (mean squared error). No noise schedule is needed.
Sampling: start from noise at t = 0 and follow the learnt velocity field to t = 1 by solving the ODE
dx/dt = v(x, t) with `nfe` Euler steps (default 32): x <- x + v(x, t) / nfe.

The network (astra.bench_net.UNet1d), its size, the batches, the optimiser and the guidance option are the
same as for the diffusion model, so the two differ only in the path from noise to data, the training
target, and the sampler.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bench_net import UNet1d, guided
from . import bench_grid
from .bench_run import Family, grid_jobs
from .classifier import autocast

DEFAULTS = dict(batch=256, lr=2e-4, base=88, t_sampling="uniform", p_uncond=0.1, grad_clip=1.0,
                nfe=32, guidance=1.0, solver="euler", chunk=20, use_phase=True)
GRID, FINAL, SPEED = bench_grid.GRID["fm"], bench_grid.FINAL["fm"], bench_grid.SPEED["fm"]   # astra/bench_grid.py
JOBS = grid_jobs("fm", GRID, FINAL)
TINY = dict(nfe=4)
LOSS_T = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)                  # times at which losses() is measured


class MODEL(Family):
    SWEEPS = ({"nfe": 8}, {"guidance": 2.0}, {"nfe": 4}, {"nfe": 16}, {"nfe": 64}, {"guidance": 1.5},
              {"solver": "midpoint"})

    def __init__(self, cfg, data, device):
        super().__init__(cfg, data, device)
        self.net = UNet1d(data["n_classes"], cfg["base"], use_phase=cfg["use_phase"]).to(device)
        self.ema = self.make_ema(self.net)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.track(net=self.net, ema=self.ema, opt=self.opt, scaler=self.scaler)

    def eval_nets(self):
        return {"ema": self.ema}

    def step(self, draw, it):
        c = self.cfg
        idx = draw()
        x1, (ph, camp, y) = self.data["u"][idx], self.cond(idx)
        x0 = torch.randn_like(x1)
        t = torch.rand(len(idx), device=self.dev) if c["t_sampling"] == "uniform" else \
            torch.sigmoid(torch.randn(len(idx), device=self.dev))
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        y = torch.where(torch.rand(len(idx), device=self.dev) < c["p_uncond"], torch.full_like(y, self.net.n_classes), y)
        with autocast(self.dev, self.amp):
            out = self.net(xt, t, ph, camp, y)
        loss = F.mse_loss(out.float(), x1 - x0)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(self.net.parameters(), c["grad_clip"])
        self.scaler.step(self.opt)
        self.scaler.update()
        self.update_ema()
        return torch.nan_to_num(loss.detach())[None]

    def velocity(self, x, t, cond, w):
        tt = torch.full((len(x),), t, device=self.dev)
        with autocast(self.dev, self.amp):
            out = guided(self.ema, x, tt, *cond, w)
        return out.float()

    def sample_batch(self, j, g, nfe=None, guidance=None, solver=None):
        """Solve dx/dt = v from t = 0 (noise) to t = 1 (data). Euler: 1 network call per step.
        Midpoint: 2 calls per step, so nfe // 2 steps for the same cost."""
        c = self.cfg
        nfe, w = nfe or c["nfe"], c["guidance"] if guidance is None else guidance
        solver, cond = solver or c["solver"], self.cond(j)
        x = torch.randn(len(j), self.data["L"], device=self.dev, generator=g)
        n = nfe if solver == "euler" else max(nfe // 2, 1)
        for i in range(n):
            t = i / n
            if solver == "euler":
                x = x + self.velocity(x, t, cond, w) / n
            else:
                half = x + self.velocity(x, t, cond, w) / (2 * n)
                x = x + self.velocity(half, t + 0.5 / n, cond, w) / n
        clip = self.data["out_scale"]
        return self.to_mag(x.clamp(-clip, clip), j)            # real curves lie in [-2, 2]

    @torch.no_grad()
    def losses(self, idx, seed):
        """The training loss of the EMA network at six fixed times, with fixed noise."""
        g = torch.Generator(device=self.dev).manual_seed(seed)
        x1, (ph, camp, y) = self.data["u"][idx], self.cond(idx)
        tot = 0.0
        for t in LOSS_T:
            x0 = torch.randn(x1.shape, device=self.dev, generator=g)
            tt = torch.full((len(idx),), t, device=self.dev)
            tot += float(F.mse_loss(self.ema((1 - t) * x0 + t * x1, tt, ph, camp, y), x1 - x0))
        return tot / len(LOSS_T)
