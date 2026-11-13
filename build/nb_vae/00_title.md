# ASTRA · Notebook 02 — Replicating PELS-VAE (Martínez-Palomera, Bloom & Abrahams 2020)

**CS787 (Generative AI) project, Group 4.** Runs on Thamilselvan's Kaggle account.

**Goal of this notebook.** Reproduce the secondary base paper as exactly as possible: *Deep Generative Modeling of Periodic Variable Stars Using Physical Parameters*, arXiv 2005.07773. We use **their own data release and their released code's settings**. Everything is re-implemented here, and our port is checked against their code, which gives the same parameter count, 300,897, and the same outputs.

| What we reproduce | Paper |
|---|---|
| Fig. 4: validation loss over training, TCN vs GRU, with/without physical parameters | ≈ 0.011 (TCN), 0.010 (TCN + physics), 0.014 (GRU) at epoch 100 |
| Figs. 5, 7: reconstructions for all 8 classes | shape and class details recovered |
| Fig. 6: the 4-D latent space | close to Gaussian, weak class clustering |
| **Table 3**: regression physical parameters → latent space, validation RMSE | cVAE: linear 0.863, RF **0.299**, MLP 0.863 · cVAE-P: 0.794, **0.289**, 0.798 |
| Fig. 8: RR Lyrae generated at increasing T_eff | sawtooth (RRab) → sinusoidal (RRc) with the RF regressor |

**The method, in the words of the VAE slides.** An **encoder** compresses a light curve (phase, magnitude, error at 600 points) plus its class into the mean and variance of a 4-D Gaussian. A latent vector is sampled with the *reparameterisation trick* (z = μ + σ·ε), and a **decoder** rebuilds the magnitudes from z, the class and the phases. The loss is reconstruction error plus β × the KL divergence to N(0, I), which keeps the latent space smooth enough to sample from. Both the encoder and the decoder are **Temporal Convolutional Networks**: 1-D *causal*, *dilated* convolutions with residual blocks, so each block sees twice as far back as the previous one.

**Paper vs released code (we follow the code, because it produced the paper's figures):**
- The loss is plain MSE + 0.75 · KL / 10⁴. The paper's Eq. 4 instead has an error-weighted MSE and a KL term on the errors; the code comments these out.
- The best model has 9 TCN blocks with kernel 5 (W&B run `b68z1hwo`). The paper's Table 2 marks 7 and 7 in bold.
- Training runs 150 epochs (the paper's text and Fig. 4 say 100).
- The split is a random 80/20 of all 65,760 curves (seed 13). The paper says 60/20/20 with real-only test stars. Because the file mixes augmented copies of the same stars, the code's split leaks copies of validation stars into training. We keep it for exactness and flag it in the analysis.
- The Table 3 MLP gets unscaled inputs (T_eff ≈ 5000), which is why it scores the same as the linear model.

**How to run (Thamilselvan's account)**
- Accelerator **GPU T4 ×2** · Internet **On**.
- First `SMOKE = True` interactively (about 10 min): tests, tiny trainings, and an estimate of each job's hours.
- Then `SMOKE = False` → **Save Version → Save & Run All**. Each GPU works down its queue (measured in session 1: 2.8 h per cVAE, 1.9 h per cVAE-P, 0.5 h for the GRU; 6.2 h in all). Each epoch is checkpointed. `SESSION_HOURS` (10.5) is the time limit of the training, not of the notebook: a job that is not finished pauses about 40 minutes before it, and the run ends when its cells are done.
- If a job is unfinished, or to refresh only the tables and figures: *Add Input* → this notebook's previous output, `SESSION` + 1, run again. Finished models are not trained again; when all five are done no worker starts, and the run only rebuilds the tables and figures (a few minutes). Check the first lines of the log: `restored N file(s)` and the jobs with `'state': 'done'`. If it says `fresh start`, the previous output is not attached: cancel the run, or everything trains again.
