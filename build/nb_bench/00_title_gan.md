# ASTRA · Notebook 03 — Benchmark, family 1 of 4: GAN

**CS787 (Generative AI) project, Group 4.** One of four notebooks (03 GAN, 04 VAE, 05 diffusion, 06 flow matching) that train four kinds of generative models for variable-star light curves **under the same conditions** and score them with the same harness.

**This notebook's model: the conditional GAN of García-Jara et al. (2022)**, the primary base paper, with its networks and losses unchanged (notebook 01 replicates the paper itself).
- The **generator** turns noise z plus a condition (the phases of the observations, the amplitude, the class) into a light curve that should look real.
- The **discriminator** (critic) sees real and generated curves with the same condition and gives a real-vs-fake score and a class prediction (AC-GAN). The loss is the Wasserstein loss with a gradient penalty (WGAN-GP); the critic is trained 5 times per generator step.
- A new curve costs **one** pass through the generator.

**Jobs.** The search runs the six configurations of `GRID["gan"]` in `astra/bench_grid.py` (learning rate 5e-5, 1e-4, 2e-4 × batch 64, 256) on split 0; the final stage runs the winner (`FINAL["gan"]`) on splits 0–4, each run fresh.
