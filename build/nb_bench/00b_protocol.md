## The benchmark protocol (identical in notebooks 03–06)

**Question.** Do diffusion and flow matching give better synthetic light curves than a GAN or a VAE for training a classifier, especially for the rare classes, when all four are tuned, trained to their best point and scored in the same way? And what does each family cost in compute?

| What | Setting | Where it comes from |
|---|---|---|
| Data | **Catalina only** (6 classes, 100 points per curve), pre-processed as in notebook 01 | García-Jara et al. 2022; one dataset by group decision (15 Oct) |
| What a model generates | the magnitudes of one curve, given a real star's phases, amplitude and class | the paper |
| Target | the amplitude-normalised curve, magnitude / amplitude, in [−2, 2]; the amplitude is given as log₁₀, standardised | ours |
| Split | over **stars**, stratified by class: 70% train, 10% validation, 20% test; five splits (seeds 0–4); the search uses split 0 | ours |
| Batches | the paper's resampling block with γ = 0.25 (rare classes drawn more often than their share) | the paper |
| Search | 6 configurations per family on split 0, scored on **validation** stars only; the best goes to the final stage | ours |
| Training length | until 4 checkpoints in a row bring no new best validation score (at least 1 h), at most 6 h in the search and 20 h in the final stage | ours |
| Checkpoints | after 5, 10, 20, 30, 45, 60, 90, 120, 180 minutes, then every hour (every 2 h after 10 h); the final stage also after 1, 2, 3, 7.5 and 15 minutes (the diffusion-type models peak within 10 minutes, then copy); each is the EMA (0.999) of the weights | EMA: the paper |
| Hours | always "hours of training on one T4 alone": iterations ÷ the configuration's measured speed, so sharing a GPU changes nothing | ours |
| Model selection | the checkpoint with the best **validation** TSTR balanced accuracy | the paper's row G, on our metric |
| Compute-matched view | the best checkpoint within 5, 10, 15, 30, 60 minutes and 2, 5, 10 h is also scored on test | ours |
| New conditions | TSTR for conditions no training star has (one star's phases, another's amplitude, same class), at the selected and at the last checkpoint | ours |
| Classifier | the paper's CNN, 15 runs × an ensemble of 5, settings of notebook 01 | the paper + notebook 01 |
| Size | generator / decoder / denoising network: about 4 M parameters each | ours |

**Scores.**
- **TSTR**: the classifier is trained on synthetic curves only and tested on real test stars.
- **Real + synthetic**: every training curve is synthetic with probability 0.25, 0.5 or 0.75; the mix is chosen on validation stars.
- **Real only**: rows A–D of the paper's Table 5 (oversampling, soft warping) on the same split. Augmentation has to beat row A.
- The headline number is **balanced accuracy**, the mean of the per-class recalls. Macro-F1, accuracy and per-class recall are reported too, and recall is also averaged over the **rare tier** (classes with fewer than 1,000 training curves: LPV, DSCT, CEP).
- Uncertainty: the ± of a row is the standard deviation over the 15 classifier runs. Differences between two rows come with a 95% interval from a paired bootstrap over test stars (1,000 resamples). Across splits: mean ± std of 3.
- Fidelity of the synthetic curves in a classifier's feature space: per-class density and precision (do they look real?), recall and coverage (do they cover the variety of the real ones?), and FID.
- **Memorisation**, next to every classifier score (ours). For each generated curve: its distance to the real curve of the star whose condition it was given. The *ratio* compares training stars with validation stars (1 = none). The *copy share* is the share of training-star curves closer to their source than 95% of the validation-star curves are to theirs: 5% = no copying, 100% = every curve a copy. Notebook 01 found late GAN checkpoints copying their rare-class training curves.

**How to run** (the steps per account are in `notebooks/RUN_INSTRUCTIONS.md`)
- Accelerator **GPU T4 ×2** · Internet **On**.
- **SMOKE** first (`SMOKE = True`, about 30 minutes): tests, a tiny search and a tiny final stage, the speed of every grid configuration, and a test of two jobs per GPU. It must end with `ALL DELIVERABLES PRESENT`; the search measures its own speeds.
- **Search** (`STAGE = "search"`, `SMOKE = False`): six jobs on `SLOTS_PER_GPU` × 2 slots. **Final** (`STAGE = "final"`, after the search's winner and its speed are set in `astra/bench_grid.py` and the notebook rebuilt): no input needed; every split trains fresh in `runs_<family>_<dataset>_final`.
- **Next session** (if the last cell lists unfinished jobs): *Add Input* → this notebook's previous version, `SESSION` + 1, Save & Run All. Finished jobs are skipped; paused ones continue.
- **The four-way comparison:** attach the final outputs of the other benchmark notebooks; the results section then adds the comparison tables.
