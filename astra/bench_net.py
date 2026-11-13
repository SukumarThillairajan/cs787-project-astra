"""The network shared by the diffusion model and the flow-matching model of the benchmark.

Both models learn a function net(x_t, t | condition) that takes a noisy light curve and says how to clean it.
They use exactly this network, with the same size, so a difference between them comes from the training
objective and the sampler, not from the architecture.

A 1-D U-Net (ours; 3.99M parameters, sized to the paper's GAN generator: 3.99M on Catalina, 3.71M on ZTF):
  - input channels: the noisy curve, and sin / cos features of the phase of every point. The points of a
    folded light curve sit at irregular phases, and phase is periodic, so the network is told each point's
    phase through periodic features, and its convolutions wrap around from the last point to the first;
  - three resolutions (L, L/2, L/4) with residual blocks and skip connections;
  - the time t, the class and the amplitude enter every block as one added embedding vector. One extra
    class row means "no class": dropping the class in training allows classifier-free guidance at sampling.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3(cin, cout, stride=1):
    return nn.Conv1d(cin, cout, 3, stride=stride, padding=1, padding_mode="circular")


def sinusoidal(t, dim):
    """The usual sinusoidal embedding of a scalar per sample."""
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), conv3(cin, cout)
        self.e = nn.Linear(emb, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), conv3(cout, cout)
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)       # every block starts as the identity
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, h, e):
        r = self.c1(F.silu(self.n1(h))) + self.e(e)[:, :, None]
        return self.skip(h) + self.c2(F.silu(self.n2(r)))


class UNet1d(nn.Module):
    def __init__(self, n_classes, base=88, n_mid=2, n_freq=4, use_phase=True):
        super().__init__()
        self.n_classes, self.base, self.n_freq = n_classes, base, n_freq
        self.use_phase = use_phase      # False = the ablation: the network is not told where the points are in phase
        b, emb = base, 4 * base
        self.t_mlp = nn.Sequential(nn.Linear(b, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.y_emb = nn.Embedding(n_classes + 1, emb)               # row n_classes = "no class"
        self.a_mlp = nn.Sequential(nn.Linear(1, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = conv3(1 + 2 * n_freq, b)
        self.enc0 = nn.ModuleList([ResBlock(b, b, emb), ResBlock(b, b, emb)])
        self.down1 = conv3(b, 2 * b, stride=2)
        self.enc1 = nn.ModuleList([ResBlock(2 * b, 2 * b, emb), ResBlock(2 * b, 2 * b, emb)])
        self.down2 = conv3(2 * b, 4 * b, stride=2)
        self.mid = nn.ModuleList([ResBlock(4 * b, 4 * b, emb) for _ in range(n_mid)])
        self.up1 = conv3(4 * b, 2 * b)
        self.dec1 = nn.ModuleList([ResBlock(4 * b, 2 * b, emb), ResBlock(2 * b, 2 * b, emb)])
        self.up0 = conv3(2 * b, b)
        self.dec0 = nn.ModuleList([ResBlock(2 * b, b, emb), ResBlock(b, b, emb)])
        self.out_norm, self.out = nn.GroupNorm(8, b), conv3(b, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, t, ph, camp, y):
        """x (B, L) noisy curve, t (B,) in [0, 1], ph (B, L) phases, camp (B,) amplitude condition,
        y (B,) class ids (n_classes = no class). Returns (B, L)."""
        e = self.t_mlp(sinusoidal(t * 1000, self.base)) + self.y_emb(y) + self.a_mlp(camp[:, None])
        k = 2 * math.pi * torch.arange(1, self.n_freq + 1, device=x.device)[None, :, None] * ph[:, None, :]
        feats = torch.cat([k.sin(), k.cos()], 1)
        if not self.use_phase:           # same network, same size, but the phase features are blanked
            feats = torch.zeros_like(feats)
        h0 = self.inp(torch.cat([x[:, None], feats], 1))
        for blk in self.enc0:
            h0 = blk(h0, e)
        h1 = self.down1(h0)
        for blk in self.enc1:
            h1 = blk(h1, e)
        h = self.down2(h1)
        for blk in self.mid:
            h = blk(h, e)
        h = torch.cat([self.up1(F.interpolate(h, size=h1.shape[-1], mode="nearest")), h1], 1)
        for blk in self.dec1:
            h = blk(h, e)
        h = torch.cat([self.up0(F.interpolate(h, size=h0.shape[-1], mode="nearest")), h0], 1)
        for blk in self.dec0:
            h = blk(h, e)
        return self.out(F.silu(self.out_norm(h)))[:, 0]


def guided(net, x, t, ph, camp, y, w):
    """Classifier-free guidance: w = 1 is the plain conditional model; w > 1 pushes the output further
    from what the model would do without knowing the class (2 network calls instead of 1)."""
    out = net(x, t, ph, camp, y)
    if w == 1.0:
        return out
    none = net(x, t, ph, camp, torch.full_like(y, net.n_classes))
    return none + w * (out - none)
