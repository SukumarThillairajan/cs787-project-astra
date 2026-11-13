## 6 · Can this VAE generate new curves? Sampling from the prior

The paper generates curves through the regression of Table 3. A VAE is also meant to generate by drawing z from the **prior** N(0, I) and decoding it, and that is how the ASTRA benchmark will use a VAE for augmentation. This only works if the encoder's codes fill the prior, which the table above measures. Here we look at it directly, for each seed of the cVAE:
- row 1: z = the latent mean of a real validation curve (a reconstruction, for reference);
- row 2: z drawn from N(0, I), the prior;
- row 3: z drawn from a Gaussian fitted to the latent means of that class (the codes the decoder was trained on).

If rows 2 and 3 look alike, the prior is safe to sample from. If row 2 shows shapes that row 3 never does, the prior has holes.
