## 1 · Data: their Zenodo release (OGLE-III I-band + Gaia DR2)

`OGLE3_lcs_I_meta_snr5_augmented_folded_trim600.npy.gz` (296 MB) is the exact file their model was trained on:
- **Light curves:** OGLE-III, with S/N ≥ 5, 3 rounds of 3σ clipping and multi-mode sub-types removed. Each is cross-matched to Gaia DR2 and validated, then phase-folded and randomly sub-sampled to 300 points.
- **Augmentation:** rare classes are topped up to 5,000 and common ones to about 10,000 curves (their Table 1, column 5). Each copy re-draws every magnitude from N(m, error), shifts the phase at random and re-samples 300 points.
- **Two cycles:** every curve is repeated over phase 0 to 2, so it has 600 points.
- **Physical parameters:** the metadata carries Gaia's T_eff, absolute G magnitude, colour, radius and luminosity, next to OGLE's period.

The table below checks the file against their Table 1 (clean-validated stars and augmented totals).
