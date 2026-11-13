## 5 · Table 3 and Fig. 8: from physical parameters to the latent space

This follows their `latent_space_visualization.ipynb`:
- Keep the curves with all three of absolute G magnitude, T_eff and period, then one row per star (43,031 → 21,796 stars).
- Regress the latent mean μ on the three raw parameters with a linear model (no intercept), a random forest (100 trees) and an MLP (50 tanh units).
- Their dummy variant adds a 4th input ~ U(0,1) and pairs each star with 10 latent samples z ~ N(μ, σ). This keeps some randomness when generating from fixed parameters.

The paper calls its Table 3 numbers validation RMSE on 20% of the data; their notebook scores the fit on the rows it was fitted on. So the table below reports three scores for every regressor, and for both seeds of each model:
- **train RMSE (in-sample)**: scored on the stars the regressor was fitted on;
- **val RMSE (new stars)**: an 80/20 split over stars, so a validation star is never seen in training;
- **val RMSE (split over curves)**: an 80/20 split over all curves, so the augmented copies of one star land on both sides, as in their VAE split.

"Predict the mean" is the RMSE of a regressor that knows nothing; a useful regressor must be clearly below it.

**Fig. 8** feeds RR Lyrae parameters (M_G = −0.24, P = 0.45 d, T_eff from 6000 to 7500 K) through each regressor and then the decoder. With the RF, the paper sees the sawtooth RRab shape turn into the sinusoidal RRc shape as T_eff rises.
