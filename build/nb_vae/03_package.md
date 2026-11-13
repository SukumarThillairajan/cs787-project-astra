## The `astra` package

The next cells write small modules to `/kaggle/working/astra/`, so the GPU workers (one process per GPU) can import them:
- `common`: paths, seeds, the time guard, safe saving;
- `pelsvae`: their data file, normalisation and split, exactly as in their `datasets.py`;
- `tcn_vae`: the TCN and GRU VAEs and their loss, ported from their `tcn.py`, `vae_models.py` and `vae_training.py`;
- `run_vae`: the per-GPU worker, which trains with checkpointing and saves latent codes and reconstructions.
