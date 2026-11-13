# ---- Self-tests: GAN ----
from astra.gan import gradient_penalty

assert N_PARAMS["G"] == bench_data.G_PARAMS[L], "the generator must be the paper's network"
cond = model.cond(idx)
u = model.G_ema(torch.randn(16, model.latent), *cond)
assert u.shape == (16, L) and float(u.abs().max()) <= data_cpu["out_scale"] + 1e-6, "tanh * 2 bounds the output"
score, logits = model.D(data_cpu["u"][idx], *cond)
assert score.shape == (16,) and logits.shape == (16, len(CLASSES))
# known answer: a critic whose gradient has norm exactly 1 everywhere gets a gradient penalty of 0
unit_critic = lambda x, *c: (x.sum(1) / L ** 0.5, None)
gp = gradient_penalty(unit_critic, data_cpu["u"][idx], u.detach(), cond)
assert float(gp) < 1e-8, float(gp)
print("GAN tests passed | generator", f"{N_PARAMS['G']:,}", "parameters (the paper's network), critic", f"{N_PARAMS['D']:,}")
del data_cpu, model
