# ---- Self-tests ----
from astra import sampling, classifier, warping, metrics, gan

y = np.array([0] * 900 + [1] * 90 + [2] * 10)
for g in (0.0, 0.25, 1.0):
    rs = sampling.ClassResampler(y, g, seed=1)
    assert np.allclose(np.bincount(y[rs.sample(100_000)], minlength=3) / 100_000, rs.p, atol=0.01)

M, N = 3, len(CLASSES)
ens = classifier.CNNEnsemble(M, N, L, dropout=0.0).train()
x = torch.randn(M, 16, 2, L)
out = ens(x)
for m in range(M):                  # every member == the same CNN built from nn.Conv1d / nn.Linear
    assert torch.allclose(classifier.reference_cnn(ens, m).train()(x[m]), out[m], atol=1e-4)
xg = torch.randn(M, 8, 2, L, requires_grad=True)
ens(xg)[0].sum().backward()
assert xg.grad[1:].abs().max() == 0, "members are independent"

ph = torch.sort(torch.rand(3, 50, L), -1).values
xb = torch.stack([torch.randn(3, 50, L), ph], 2)
for mode in ("time", "mag", "mixed"):
    w = warping.soft_warp(xb, mode)
    assert (torch.diff(w[..., 1, :], dim=-1) >= -1e-6).all() and w[..., 1, :].max() <= 1

a = torch.randn(2000, 16)
assert abs(metrics.prdc(a, a.clone())["density"] - 1.2) < 1e-4                 # a copy: (k+1)/k
same = metrics.prdc(a, torch.randn(2000, 16))
assert abs(same["density"] - 1) < 0.15 and same["recall"] > 0.8
collapsed = metrics.prdc(a, 0.05 * torch.randn(2000, 16))
assert collapsed["recall"] < 0.2, "a collapsed generator has low recall"

data_cpu = gan.load_data(NPZ, "cpu")
G = gan.Generator(LATENT, L, N, data_cpu["scale"]); D = gan.Discriminator(L, N)
j = data_cpu["train_idx"][:8]
cond = (data_cpu["ph"][j], data_cpu["amp"][j], data_cpu["onehot"][j])
xg = G(torch.randn(8, LATENT), *cond)
assert xg.shape == (8, L)
gan.gradient_penalty(D, data_cpu["x"][j], xg.detach(), cond).backward()
# the resume path on this device (reported, not asserted: a resumed job continues either way)
from astra.common import restore_rng, torch_save
torch_save({"rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if DEV == "cuda" else None}, WORK / "rng_test.pt")
first = torch.rand(3, device=DEV)
ok = restore_rng(torch.load(WORK / "rng_test.pt", map_location=DEV, weights_only=False), DEV)
print("resume check: random-number state", "restored" if ok and torch.equal(first, torch.rand(3, device=DEV))
      else "NOT restored (a resumed GAN continues with fresh random numbers)")
(WORK / "rng_test.pt").unlink()
print(f"all self-tests passed | G {sum(p.numel() for p in G.parameters()) / 1e6:.2f}M params, "
      f"D {sum(p.numel() for p in D.parameters()) / 1e6:.2f}M | tanh scale s = {data_cpu['scale']:.2f} mag")
del data_cpu, G, D, ens
