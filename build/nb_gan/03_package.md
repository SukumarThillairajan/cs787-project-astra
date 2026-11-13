## The `astra` package

The code lives in small modules written to `/kaggle/working/astra/` by the next cells, so that the GPU workers (separate processes, one per GPU) can import it:
- `common`: paths, seeds, the time guard, safe saving;
- `catalina`, `ztf`: the two datasets (download and pre-processing);
- `sampling`: the γ-resampling block;
- `classifier`: the paper's CNN, trained as 75 independent copies at once;
- `warping`: the soft-warping baselines;
- `metrics`: density, recall, G-score, FID, Welch's test;
- `gan`: generator, discriminator, training loop, checkpoint evaluation and model selection;
- `run_gan`: the per-GPU worker.
