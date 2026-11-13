# ASTRA · Notebook 06 — Benchmark, family 4 of 4: flow matching

**CS787 (Generative AI) project, Group 4.** One of four notebooks (03 GAN, 04 VAE, 05 diffusion, 06 flow matching) that train four kinds of generative models for variable-star light curves **under the same conditions** and score them with the same harness.

**This notebook's model: conditional flow matching** (Lipman et al. 2023) with straight paths (rectified flow, Liu et al. 2023). Flow matching is not in the course syllabus, so here is the idea in full:
- **Paths.** Draw a noise curve x_0 ~ N(0, I) and a real curve x_1, and join them with a straight line: x_t = (1 − t) · x_0 + t · x_1, for t from 0 to 1. A point that moves along this line has the constant velocity x_1 − x_0.
- **Training.** Pick t at random and train a network v(x_t, t, condition) to predict that velocity (mean squared error). The condition is the same as everywhere: phases, amplitude, class. There is no noise schedule to choose.
- **Sampling.** Start from noise at t = 0 and follow the learnt velocities to t = 1, in small Euler steps: x ← x + v(x, t)/n. A new curve costs n = 32 network passes here.
- Compared with diffusion (notebook 05): the same network, the same size, the same batches. Only the path from noise to data (a straight line instead of the diffusion's curved one), the training target (velocity instead of noise) and the sampler (an ODE solver) differ. Straighter paths should need fewer sampling steps; the sampling sweep at the end tests this.

**Jobs.** The search runs the six configurations of `GRID["fm"]` in `astra/bench_grid.py` (uniform or logit-normal training times × learning rate 1e-4, 2e-4, 4e-4) on split 0; the final stage runs the winner (`FINAL["fm"]`) on splits 0–4, each run fresh, and sweeps solver, steps and guidance at its best checkpoint.
