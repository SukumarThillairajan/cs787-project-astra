# ASTRA · Notebook 04 — Benchmark, family 2 of 4: VAE

**CS787 (Generative AI) project, Group 4.** One of four notebooks (03 GAN, 04 VAE, 05 diffusion, 06 flow matching) that train four kinds of generative models for variable-star light curves **under the same conditions** and score them with the same harness.

**This notebook's model: a conditional VAE built from the GAN's own two networks.**
- The **encoder** compresses a light curve and its condition (phases, amplitude, class) into the mean and variance of a Gaussian over a small latent vector z. It is the GAN's critic with a new last layer.
- A latent vector is sampled with the *reparameterisation trick* (z = μ + σ·ε), and the **decoder** rebuilds the curve from z and the condition. It is the GAN's generator, with the same inputs and the same size, plus one output: the noise level of the curve.
- The loss is the negative ELBO: reconstruction error plus β × the KL divergence between the encoder's Gaussian and the prior N(0, I), with β = 1.
- A new curve: draw z from the prior, decode it (one pass), add noise at the predicted level.

So the VAE and the GAN of notebook 03 share the generator and differ only in how it is trained. This is **not** PELS-VAE (notebook 02): that model weights the KL term by 10⁻⁴ and is never sampled from its prior. Our replication found its latent codes far from N(0, I), so here the KL term has full weight and the model is judged on curves sampled from the prior.

**Jobs.** The search runs the eight configurations of `GRID["vae"]` in `astra/bench_grid.py` (β = 0.1, 1, 4, 16 × learning rate 2e-4, 1e-3; β = 16 added for the search's second session, because the score rose with β up to the grid's edge) on split 0; the final stage runs the winner (`FINAL["vae"]`) on splits 0–4, each run fresh.
