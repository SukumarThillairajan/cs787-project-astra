## 2 · Self-tests

- Our TCN-VAE with their best settings has **300,897 parameters**, the number their W&B run reports. The GRU and the physics-conditioned versions build too.
- The split reproduces their procedure: 80/20 of all rows with seed 13, and 20% of the 43,031 rows that have all three physical parameters for cVAE-P.
- The encoder, the reparameterisation and the decoder give the right shapes, and the loss is positive.
- Their cosine schedule, stepped with the epoch number, goes down, up and down again over 150 epochs (plotted).
