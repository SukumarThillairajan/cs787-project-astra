# ---- Self-tests: diffusion ----
ratio = N_PARAMS["net"] / bench_data.G_PARAMS[L]
assert abs(ratio - 1) < 0.10, f"the denoising network should match the GAN generator's size (ratio {ratio:.2f})"
ab = model.ab
assert ab[0] > 0.99 and ab[-1] < 1e-3 and (ab[1:] < ab[:-1]).all(), "signal share must fall from ~1 to ~0"
u0 = data_cpu["u"][idx]


class Oracle:                                 # a "network" that always returns the true noise for the curves u0
    n_classes = len(CLASSES)

    def __call__(self, x, tt, ph, camp, y):
        a = ab[(tt * model.T).round().long()][:, None]
        return (x - a.sqrt() * u0) / (1 - a).sqrt()


# known answer: with the true noise, the DDIM sampler must return the clean curves, for any number of steps
net, model.ema = model.ema, Oracle()
for nfe in (2, 8, 32):
    assert torch.allclose(model.sample(idx, seed=1, nfe=nfe), data_cpu["x"][idx], atol=2e-3), nfe
model.ema = net
print(f"diffusion tests passed | denoising network {N_PARAMS['net']:,} parameters "
      f"({100 * ratio:.0f}% of the GAN generator's {bench_data.G_PARAMS[L]:,})")
del data_cpu, model
