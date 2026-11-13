## 4 · Training: the workers

Every GPU runs `SLOTS_PER_GPU` workers (`python -m astra.bench_run`), each working down its own queue:
- a model job trains one configuration and takes a checkpoint after 5, 10, 20, 30, 45, 60, 90, 120, 180 minutes, then every hour (every 2 hours after 10 h; the final stage also after 1, 2, 3, 7.5 and 15 minutes), counted in hours of training on one T4 alone. At each checkpoint it generates 20,000 class-balanced curves and measures fidelity, memorisation (ratio and copy share), the model's own loss on training and validation stars, and the validation TSTR balanced accuracy (3 runs × 5 CNNs);
- it stops when 4 checkpoints in a row bring no new best validation score (after at least 1 h), or at the cap: 6 h in the search, 20 h in the final stage;
- **search stage**: the run ends with its best validation score and keeps its state;
- **final stage**: a fresh run per split (in its own folder) gets the test scores of its best checkpoint: one synthetic curve per training condition, TSTR and real + synthetic (15 runs × 5 CNNs each), the other sampling settings, the TSTR test score of the best checkpoint within 5, 10, 15, 30, 60 minutes and 2, 5, 10 h, and TSTR for conditions no training star has, at the selected and at the last checkpoint;
- a `real_s<k>` job trains the reference classifiers on the real training curves of split k (rows A–D).

In a real run, a short *canary* first runs a tiny version of each slot's first model job, so a broken setup fails early. The cell prints a status line per slot every 10 minutes, returns as soon as all workers are done, and stops a worker that is still running at the time limit (its saved state is kept). Full logs: `runs_*/<job>/log.txt`.
