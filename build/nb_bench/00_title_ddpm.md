# ASTRA · Notebook 05 — Benchmark, family 3 of 4: diffusion (DDPM)

**CS787 (Generative AI) project, Group 4.** One of four notebooks (03 GAN, 04 VAE, 05 diffusion, 06 flow matching) that train four kinds of generative models for variable-star light curves **under the same conditions** and score them with the same harness.

**This notebook's model: a conditional denoising diffusion model** (DDPM, Ho et al. 2020). Diffusion is not in the course syllabus, so here is the idea in full:
- **Forward process (no learning).** Take a real light curve and add a little Gaussian noise, 1000 times. After the last step nothing of the curve is left: it is pure noise. The noisy curve at any step t can be written in one line: x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε, where ε is the noise and ᾱ_t falls from 1 to 0.
- **Training.** Pick a real curve and a step t at random, make x_t, and train a network to predict the noise ε that was added (mean squared error). The network also sees the condition: phases, amplitude, class.
- **Sampling.** Start from pure noise and remove the predicted noise step by step. We use the DDIM sampler, which can skip steps: a new curve costs 32 network passes here instead of 1000.
- Compared with a VAE, there is no encoder and no small latent vector: the "latent" is a noise curve of the same size as the data. Compared with a GAN, there is no second network to fight: the loss is a plain regression.

**Jobs.** The search runs the six configurations of `GRID["ddpm"]` in `astra/bench_grid.py` (ε- or v-prediction × learning rate 1e-4, 2e-4, 4e-4) on split 0; the final stage runs the winner (`FINAL["ddpm"]`) on splits 0–4, each run fresh, and sweeps the sampling steps and guidance at its best checkpoint.
