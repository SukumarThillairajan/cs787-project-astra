## 7 · Observations

Written from session 1 (real run, 4 October 2026, T4 ×2, all five models finished in 6.2 h) and session 2 (11 October 2026, 5 minutes: the results cells only, on the saved models). The paper's Fig. 4 values are read off a plot (about ±0.001). The tables, figures and executed notebooks of both sessions were read.

**Replication verdict.**

| Quantity | Paper | Ours, seed 13 | Ours, seed 14 |
|---|---|---|---|
| Validation loss at epoch 100, cVAE (TCN) | ≈ 0.011 | 0.0228 | **0.0114** |
| Validation loss at epoch 100, cVAE-P (TCN + physics) | ≈ 0.010 | 0.0181 | 0.0140 |
| Validation loss at epoch 100, GRU | ≈ 0.014 | 0.0184 | not run |
| Table 3, cVAE: linear / RF / MLP | 0.863 / 0.299 / 0.863 | 0.880 / 0.893 / 0.884 (new stars) | 0.800 / 0.832 / 0.802 (new stars) |
| Table 3, cVAE-P: linear / RF / MLP | 0.794 / 0.289 / 0.798 | 0.898 / 0.924 / 0.963 (new stars) | 0.855 / 0.866 / 0.853 (new stars) |
| Table 3, predicting the mean latent vector (cVAE / cVAE-P) | not given | 0.883 / 0.901 | 0.802 / 0.852 |

- The cVAE's validation loss is reproduced by one seed of two. Seed 14 gives 0.0114 at epoch 100 and 0.0106 at epoch 150. Seed 13 has its lowest value, 0.0119, at epoch 28 and then gets worse: stopped there it would also be close to the paper.
- The cVAE-P and the GRU end 30–80% above the paper's values in both seeds we have.
- The paper's order "physics < plain TCN < GRU" is not reproduced: our cVAE-P is above our cVAE at epoch 100 in seed 14. The two are trained on different curves (43,031 against 65,760), so their losses are not strictly comparable, in the paper as well as here.
- Table 3: for the cVAE, the linear model and the MLP match the paper on new stars (within 2.5%); for the cVAE-P they are 13% and 21% higher. The random forest matches the paper only **in-sample** (0.330 for the cVAE, 0.344 for the cVAE-P, against 0.299 and 0.289). On stars it has not seen it scores 0.893 and 0.924. In-sample it is 0.315 and 0.322 for the two models of seed 14, so this repeats in all four.
- No regressor beats knowing nothing. On new stars the best regressor and "predict the mean latent vector" score 0.880 and 0.883 (cVAE, seed 13), 0.800 and 0.802 (cVAE, seed 14), 0.898 and 0.901 (cVAE-P, seed 13), 0.853 and 0.852 (cVAE-P, seed 14). The three physical parameters carry almost no information about the latent vector of a star the regressor has not seen.
- The other explanation of the paper's 0.299, a leak through the augmented copies, does not hold. With the split over curves, 62.1% of the validation curves have a copy of their star in training, and the random forest still scores 0.83, 0.90, 0.83 and 0.79 in the four models. Only scoring on the training rows gives a number near the paper's.
- Fig. 8: with the random forest we get a clear sawtooth (RRab) at every T_eff from 6000 to 7500 K, slightly lower in amplitude at the hot end. The change to a sinusoidal RRc shape that the paper shows does not appear. The linear and MLP rows barely react to T_eff. Seed 14 (`fig8_rrlyr_teff_tcn_s14.png`) agrees: its random-forest row is a sawtooth at every T_eff and only jumps in phase between 6750 and 7125 K, and its linear and MLP rows give the same symmetric curve at all five temperatures. Neither seed shows the paper's change of shape.

**Paper vs code.** Following the released code has a visible cost. Its cosine schedule (T_max = 50, stepped with the epoch) brings the learning rate back up to 10⁻³ between epochs 50 and 100. In seed 13 the validation loss starts to rise exactly there (0.0129 at epoch 50, 0.0142 at 70, 0.0228 at 100, 0.0331 at 150) while the training loss keeps falling (0.0141 → 0.0118). The paper's text and Fig. 4 stop at epoch 100; the code trains to 150 and keeps the last weights, with no early stopping.

**Over/under-fitting.** No model under-fits: every training loss falls for the whole run.
- cVAE, seed 13: strong over-fitting after epoch 60 (validation 2.8 × training at the end). Its reconstructions of validation curves have the right shape but are shifted in phase by up to about 0.1 cycle.
- cVAE, seed 14: none. Validation (0.0106) stays below training (0.0128); training is measured with dropout on, validation with dropout off. Its reconstructions follow the validation curves in phase and in shape (`recon_tcn_s14.png`), where seed 13's are shifted.
- cVAE-P: mild in both seeds (lowest validation loss 0.0139 and 0.0135 at epochs 22 and 27, then 0.0178 and 0.0154 at the end).
- Four of the five models have their lowest validation loss between epochs 19 and 28, before the learning rate comes back up; only the cVAE of seed 14 is best at the last epoch.
- GRU: validation between 0.016 and 0.019 and noisy, training still falling.
- Seeds 13 and 14 share the split and differ only in the initial weights and the batch order. So the outcome of the released recipe depends on the seed, and a single run (as in the paper) cannot show that. Why seed 13 over-fits is not established: the phase shifts suggest the encoder memorises where each training curve starts, which cannot carry over to the augmented copies, because those are shifted in phase at random (our reading, not tested).

**Leakage in their split.** 47.5% of the validation stars also have a copy in the training set. The validation loss is therefore not a test on new stars, and the copies are no protection either: seed 13 over-fits although half of its validation stars are "known". A split over stars is needed before this loss can be called generalisation; the ASTRA benchmark splits over stars.

**Latent space.** The latent codes are far from the N(0, I) prior. For the cVAE (seed 13) the means of μ are −0.40, −0.23, −0.12, 0.08 and its spreads 0.38, 0.98, 0.91, 0.78; the first two dimensions are strongly correlated (Fig. 6); the encoder's own σ is 0.02–0.05. With the KL term divided by 10⁴ the model is in practice an autoencoder: each curve gets an almost exact code, and nothing pulls the codes towards the prior. Drawing z from N(0, I) will often land where the decoder was never trained. Session 2 measures it: of 100,000 draws from the prior, 78% fall outside the region that holds 99% of the codes for the cVAE of seed 13, and practically all (99.998%) for seed 14, where one latent dimension has collapsed (the spread of its μ is 0.001). For the cVAE-P it is 78% and 65%, for the GRU 40%. The curves decoded from prior draws (`prior_samples_tcn.png`, `prior_samples_tcn_s14.png`) are smooth and periodic, so the decoder does not break outside the region of the codes. What is lost is variety and class shape. In seed 14 the four prior draws of a class are one template shifted in phase: the same sawtooth for the Cepheids and for the RR Lyrae, and for the eclipsing binaries a sinusoid-like curve without the narrow eclipses that the encoded real curves have. In seed 13 the RR Lyrae draws lose their steep rise and look like the LPV draws. A Gaussian fitted to the class's own codes gives somewhat more varied shapes. So most of what the latent vector carries is where the curve starts in phase. That is a product of the augmentation (random phase shifts), not a property of the star, and it fits Table 3: physical parameters cannot predict it. Consequence for the benchmark: its VAE keeps the full KL term (β = 1) and is judged on curves sampled from the prior, not on reconstructions.

**TCN vs GRU; with vs without physics.** The TCN (seed 14) is clearly better than the GRU (0.0114 against 0.0184 at epoch 100), as in the paper, but the GRU has 46 k parameters against 301 k and trains 6 × faster (0.18 against 1.11 minutes per epoch). The cVAE-P reconstructions follow the data closely, with two visible failures: deep, narrow eclipses come out far too shallow, and gaps in phase are bridged by straight segments. The first is the over-smoothing of sharp features that ASTRA expects from a VAE.

**What this means for ASTRA.**
1. PELS-VAE as released is a good compressor, not yet a generator: it is never sampled from its prior in the paper, and its codes do not fill the prior.
2. Its numbers need more than one seed and a split over stars; ours has two seeds and their split.
3. Table 3's random-forest result is a memorisation effect on our data, so "physical parameters predict the latent vector" is not supported by this replication.

