"""Offline unit tests for the GAN-notebook package (CPU). Run from the project root:
    python build/tests/test_gan_units.py <data_dir>
<data_dir> holds CatalinaVars.tbl and AllVar.phot.gz; outputs go to $ASTRA_WORK.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astra import catalina, classifier, gan, metrics, sampling, warping  # noqa: E402
from astra.common import WORK  # noqa: E402

torch.manual_seed(0)
data_dir = Path(sys.argv[1])
npz = WORK / "data" / "catalina_L100.npz"

# 1. pre-processing (real data) ---------------------------------------------------------------
if not npz.exists():
    t = time.time()
    out, table = catalina.preprocess(data_dir, L=100, class_range_k=2.0, seed=0)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **out)
    print(table.to_string(index=False), f"\n({time.time() - t:.0f} s)")
z = np.load(npz)
assert z["phase"].shape[1] == 100 and (np.diff(z["phase"], axis=1) >= 0).all(), "phases sorted"
assert np.allclose(np.median(z["mag"], 1), 0, atol=1e-5), "median centred"
assert set(np.unique(z["split"])) == {0, 1, 2}
nv = np.bincount(z["y"][z["split"] == 1]); nt = np.bincount(z["y"][z["split"] == 2])
assert (nv == nv[0]).all() and (nt == nt[0]).all(), "val/test balanced"
print("1 preprocessing ok: val/test per class", nv[0], "| train", (z["split"] == 0).sum())

# 2. resampler ---------------------------------------------------------------------------------
y = np.array([0] * 900 + [1] * 90 + [2] * 10)
for g in (0.0, 0.25, 1.0):
    rs = sampling.ClassResampler(y, g, seed=1)
    fr = np.bincount(y[rs.sample(200_000)], minlength=3) / 200_000
    assert np.allclose(fr, rs.p, atol=0.01), (g, fr, rs.p)
rs = sampling.ClassResampler(y, 1.0, seed=1)
first = rs.sample(30)
assert len(set(first[y[first] == 2])) == min(10, (y[first] == 2).sum()), "no repeats before a queue is used up"
print("2 resampler ok: p(gamma=.25) =", np.round(sampling.ClassResampler(y, .25).p, 3))

# 3. grouped CNN ensemble == separate standard CNNs ------------------------------------------------
M, N, L = 3, 6, 100
ens = classifier.CNNEnsemble(M, N, L, dropout=0.0).train()        # train mode: BatchNorm uses batch statistics
x = torch.randn(M, 16, 2, L)
out = ens(x)
assert out.shape == (M, 16, N)
for m in range(M):
    ref = classifier.reference_cnn(ens, m).train()
    assert torch.allclose(ref(x[m]), out[m], atol=1e-4), (m, (ref(x[m]) - out[m]).abs().max())
    assert torch.allclose(ref[:20](x[m]).flatten(1), ens.features(x)[m], atol=1e-4)
xg = torch.randn(M, 8, 2, L, requires_grad=True)
ens(xg)[0].sum().backward()
assert xg.grad[1:].abs().max() == 0, "member 0 must not see other members' inputs"
p = classifier.predict_proba(ens, torch.randn(10, 2, L))
assert p.shape == (M, 10, N) and torch.allclose(p.sum(-1), torch.ones(M, 10), atol=1e-5)
print("3 CNN ensemble ok: identical to standard nn.Conv1d/nn.Linear CNNs, members independent; features dim",
      ens.features(x).shape[-1])

# 4. warping ----------------------------------------------------------------------------------------
ph = torch.sort(torch.rand(3, 50, L), -1).values
xm = torch.randn(3, 50, L)
xb = torch.stack([xm, ph], 2)
for mode in ("time", "mag", "mixed"):
    w = warping.soft_warp(xb, mode)
    assert (torch.diff(w[..., 1, :], dim=-1) >= -1e-6).all(), "phase order kept"
    assert w[..., 1, :].min() >= 0 and w[..., 1, :].max() <= 1
    assert w[..., 0, :].min() >= xm.min() - 1e-6 and w[..., 0, :].max() <= xm.max() + 1e-6
    changed = (w - xb).abs().amax((-1, -2)) > 0
    assert changed.float().mean() > 0.5, mode
print("4 warping ok")

# 5. metrics -----------------------------------------------------------------------------------------
a = torch.randn(2000, 16)
copy = metrics.prdc(a, a.clone(), k=5)
assert copy["recall"] == 1 and copy["precision"] == 1 and copy["coverage"] == 1
assert abs(copy["density"] - 6 / 5) < 1e-4, copy          # each copy also sits in its twin's ball: (k+1)/k
same = metrics.prdc(a, torch.randn(2000, 16), k=5)        # same distribution, independent sample
assert abs(same["density"] - 1) < 0.15 and same["recall"] > 0.8 and same["precision"] > 0.8, same
far = metrics.prdc(a, a + 100, k=5)
assert far["recall"] == 0 and far["density"] == 0
narrow = metrics.prdc(a, 0.05 * torch.randn(2000, 16), k=5)      # a collapsed generator
assert narrow["recall"] < 0.2 and narrow["density"] > 1, narrow
g = metrics.g_score([[0.2, 1.0], [0.8, 0.5], [0.5, 0.9]], [[0.1, 0.2], [0.9, 0.8], [0.5, 0.3]])
assert np.argmax(g) == 1
f0 = metrics.fid(a.numpy().astype(float), a.numpy().astype(float))
assert abs(f0) < 1e-3
print("5 metrics ok: collapsed generator ->", {k: round(v, 3) for k, v in narrow.items()})

# 6. GAN shapes, gradient penalty, one training step ------------------------------------------------
data = gan.load_data(npz, "cpu")
G = gan.Generator(16, L, data["n_classes"], data["scale"])
D = gan.Discriminator(L, data["n_classes"])
nG, nD = sum(p.numel() for p in G.parameters()), sum(p.numel() for p in D.parameters())
j = data["train_idx"][:8]
cond = (data["ph"][j], data["amp"][j], data["onehot"][j])
xg = G(torch.randn(8, 16), *cond)
assert xg.shape == (8, L) and xg.abs().max() <= data["scale"] + 1e-5
d, logit = D(data["x"][j], *cond)
assert d.shape == (8,) and logit.shape == (8, data["n_classes"])
gp = gan.gradient_penalty(D, data["x"][j], xg.detach(), cond)
gp.backward()
assert all(p.grad is not None for n, p in D.named_parameters() if "weight" in n)   # biases do not enter dD/dx
print(f"6 GAN ok: G {nG / 1e6:.2f}M params, D {nD / 1e6:.2f}M params, tanh scale s = {data['scale']:.2f} mag")
print("ALL UNIT TESTS PASSED")
