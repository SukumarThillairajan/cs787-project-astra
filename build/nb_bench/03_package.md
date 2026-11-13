## The `astra` package

The code lives in small modules written to `/kaggle/working/astra/` by the next cells, so that the GPU workers (separate processes, one per GPU) can import it. The first group is shared with notebook 01 and unchanged; the `bench_*` modules are the benchmark.
- `common`, `catalina`, `ztf`, `sampling`, `classifier`, `warping`, `metrics`: paths and the time guard, the two datasets, the γ-resampling block, the paper's CNN (75 copies trained at once), the soft-warping baselines, density / recall / FID.
- `bench_data`: the split over stars and the amplitude-normalised target.
- `bench_eval`: **the harness**, the same file in all four notebooks: the classifier fits, balanced accuracy, fidelity, the memorisation ratio, the evaluation of a checkpoint and the final test scores.
- `bench_run`: the trainer with the time budget, checkpoints, resume and model selection, and the per-GPU worker.
@@MODULES@@
- `bench_report`: tables, paired bootstrap and figures.
