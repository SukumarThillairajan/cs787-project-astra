"""Benchmark family 2 — VAE: a conditional VAE built from the GAN's own two networks (ours).

  - decoder p(x | z, condition) = the GAN's generator, with the same inputs (z, phases, amplitude, class) and
    the same size. It gives the mean curve, and one extra output: the noise level sigma of that curve;
  - encoder q(z | x, condition) = the GAN's critic, with its last layer replaced by the mean and log-variance
    of a Gaussian over z (same latent size as the GAN's noise).
So the VAE and the GAN differ in how the generator is trained, not in the generator.

Loss = minus the ELBO: -log N(x; mean, sigma^2) summed over the points + beta * KL(q(z | x) || N(0, I)),
with beta = 1 (the true bound) and the KL weight raised linearly over the first 10% of training, so the
decoder does not learn to ignore z. Why sigma is learnt: a decoder that returns only the mean gives smooth,
noise-free curves, and a classifier trained on those meets real, noisy curves at test time. A new curve is
z ~ N(0, I) -> mean, sigma -> mean + sigma * noise. Sampling without the noise is scored as a sweep.

This is not PELS-VAE: that model divides the KL term by 10^4 and is never sampled from its prior (see the
replication notebook 02); here the KL term has full weight and the model is judged on prior samples.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bench_data import LATENT
from . import bench_grid
from .bench_run import Family, grid_jobs
from .classifier import autocast
from .gan import Discriminator, Generator

DEFAULTS = dict(latent=None, batch=256, lr=2e-4, beta=1.0, kl_warmup_min=12.0, grad_clip=10.0, chunk=20)
GRID, FINAL, SPEED = bench_grid.GRID["vae"], bench_grid.FINAL["vae"], bench_grid.SPEED["vae"]   # astra/bench_grid.py
JOBS = grid_jobs("vae", GRID, FINAL)
TINY = dict(kl_warmup_min=0.1)


class Encoder(Discriminator):
    """The critic's five conv blocks; the last layer gives mean and log-variance of q(z | x, condition)."""

    def __init__(self, L, n_classes, latent):
        super().__init__(L, n_classes)
        self.fc = nn.Linear(self.fc.in_features, 2 * latent)

    def forward(self, x, ph, amp, onehot):
        L = x.shape[1]
        h = torch.cat([x[:, None], ph[:, None], amp[:, None, None].expand(-1, 1, L),
                       onehot[:, :, None].expand(-1, -1, L)], 1)
        for layer in self.conv:
            h = F.leaky_relu(layer(h), 0.2)
        return self.fc(h.flatten(1)).chunk(2, dim=1)


class Decoder(Generator):
    """The generator, plus one number per curve: log sigma, the noise level of its points."""

    def __init__(self, latent, L, n_classes, scale):
        super().__init__(latent, L, n_classes, scale)
        self.logsig = nn.Linear(4 * 1024, 1)
        nn.init.zeros_(self.logsig.weight)
        nn.init.constant_(self.logsig.bias, math.log(0.2))

    def forward(self, z, ph, amp, onehot):
        h = F.relu(self.fc(torch.cat([z, ph, amp[:, None], onehot], 1)))
        logsig = self.logsig(h).clamp(-5.0, 1.0)
        h = h.view(-1, 1024, 4)
        for i, layer in enumerate(self.deconv):
            h = layer(h)
            if i < len(self.deconv) - 1:
                h = F.relu(h)
        return (torch.tanh(h) * self.scale)[:, 0, :self.L], logsig


class MODEL(Family):
    STATS = ("loss", "nll_per_point", "kl", "sigma")
    SWEEPS = ({"noise": False},        # the decoder's mean only: the smooth, "blurry" VAE output
              {"temp": 0.7})           # z from a narrower Gaussian

    def __init__(self, cfg, data, device):
        super().__init__(cfg, data, device)
        N, L = data["n_classes"], data["L"]
        self.latent = cfg.get("latent") or LATENT.get(L, 16)
        self.enc = Encoder(L, N, self.latent).to(device)
        self.dec = Decoder(self.latent, L, N, data["out_scale"]).to(device)
        self.enc_ema, self.dec_ema = self.make_ema(self.enc), self.make_ema(self.dec)
        self.opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=cfg["lr"])
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.track(enc=self.enc, dec=self.dec, enc_ema=self.enc_ema, dec_ema=self.dec_ema, opt=self.opt,
                   scaler=self.scaler)

    def eval_nets(self):
        return {"dec_ema": self.dec_ema, "enc_ema": self.enc_ema}

    def cond(self, idx):
        d = self.data
        return d["ph"][idx], d["camp"][idx], d["onehot"][idx]

    def elbo_terms(self, enc, dec, idx, sample_z=True):
        """Per-curve -log p(x | z) (summed over the points) and KL, averaged over the batch; mean sigma."""
        x, cond = self.data["u"][idx], self.cond(idx)
        with autocast(self.dev, self.amp):
            mu, logvar = enc(x, *cond)
        mu, logvar = mu.float(), logvar.float().clamp(-12.0, 6.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample_z else mu      # reparameterisation
        with autocast(self.dev, self.amp):
            mean, logsig = dec(z, *cond)
        mean, logsig = mean.float(), logsig.float()
        nll = (0.5 * ((x - mean) / logsig.exp()) ** 2 + logsig + 0.5 * math.log(2 * math.pi)).sum(1).mean()
        kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(1).mean()
        return nll, kl, logsig.exp().mean()

    def step(self, draw, it):
        c = self.cfg
        nll, kl, sig = self.elbo_terms(self.enc, self.dec, draw())
        warm = c.get("kl_warmup_its", 0)    # the KL weight rises linearly over the first kl_warmup_min minutes
        beta = c["beta"] * min(1.0, it / warm) if warm > 0 else c["beta"]
        loss = nll + beta * kl
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(list(self.enc.parameters()) + list(self.dec.parameters()), c["grad_clip"])
        self.scaler.step(self.opt)
        self.scaler.update()
        self.update_ema()
        return torch.nan_to_num(torch.stack([loss.detach(), nll.detach() / self.data["L"], kl.detach(), sig.detach()]))

    def sample_batch(self, j, g, noise=True, temp=1.0):
        z = temp * torch.randn(len(j), self.latent, device=self.dev, generator=g)
        mean, logsig = self.dec_ema(z, *self.cond(j))
        if noise:
            mean = mean + logsig.exp() * torch.randn(mean.shape, device=self.dev, generator=g)
        return self.to_mag(mean, j)

    @torch.no_grad()
    def losses(self, idx, seed):
        """Minus the ELBO per point (beta = 1, z = the posterior mean), with the EMA networks."""
        nll, kl, _ = self.elbo_terms(self.enc_ema, self.dec_ema, idx, sample_z=False)
        return float((nll + kl) / self.data["L"])
