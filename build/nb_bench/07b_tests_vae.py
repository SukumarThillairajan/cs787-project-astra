# ---- Self-tests: VAE ----
import math

assert N_PARAMS["dec"] == bench_data.G_PARAMS[L] + 4 * 1024 + 1, "decoder = the paper's generator + the sigma output"


class PriorEncoder(torch.nn.Module):          # an encoder that always answers "the prior": mean 0, log-variance 0
    def forward(self, x, *c):
        z = torch.zeros(len(x), model.latent)
        return z, z


class FlatDecoder(torch.nn.Module):           # a decoder that always answers: mean curve 0, sigma 1
    def forward(self, z, *c):
        return torch.zeros(len(z), L), torch.zeros(len(z), 1)


# known answer: KL(N(0, I) || N(0, I)) = 0, and -log N(u; 0, 1) summed over the points
nll, kl, sig = model.elbo_terms(PriorEncoder(), FlatDecoder(), idx)
u = data_cpu["u"][idx]
assert abs(float(kl)) < 1e-6 and float(sig) == 1.0
assert torch.isclose(nll, (0.5 * u ** 2 + 0.5 * math.log(2 * math.pi)).sum(1).mean(), atol=1e-3)
mean_only = model.sample(idx, seed=3, noise=False)
assert not torch.equal(mean_only, x1), "sampling adds noise at the predicted level unless noise=False"
print("VAE tests passed | decoder", f"{N_PARAMS['dec']:,}", "parameters (the GAN's generator + 1 output), encoder",
      f"{N_PARAMS['enc']:,}", "| latent size", model.latent)
del data_cpu, model
