# ASTRA · Notebook 01 — Replicating García-Jara et al. (2022): a conditional GAN for light-curve augmentation

**CS787 (Generative AI) project, Group 4.** Catalina runs on Tharun's Kaggle account, ZTF on Thamilselvan's.

**Goal of this notebook.** Reproduce Table 5 of the primary paper, as exactly as their text allows. `DATASET` picks the column: `"catalina"` (default) or `"ztf"`. The paper:
García-Jara, Protopapas & Estévez, *Improving Astronomical Time-series Classification via Data Augmentation with Generative Adversarial Networks*, ApJ 2022 (arXiv 2205.06758).

| Row | Training set of the classifier | Paper, Catalina (test accuracy %) | Paper, ZTF |
|---|---|---|---|
| A | real, balanced by oversampling (TRTR) | 73.44 ± 1.22 | 72.61 ± 0.69 |
| B / C / D | real + soft time / magnitude / mixed warping | 74.06 / 73.64 / 73.82 | 72.69 / 72.45 / 72.53 |
| E / F | u-GAN (balanced GAN training), chosen by val. accuracy / G-score (TSTR) | 75.97 / 76.28 | 74.17 / 73.79 |
| G / H | γ-GAN (γ = 0.25), chosen by val. accuracy / G-score (TSTR) | 76.86 / **76.97 ± 0.79** | 74.37 / **74.94 ± 0.44** |

TRTR = train on real, test on real. TSTR = train on synthetic *only*, test on real. Every row is 15 independent runs of an ensemble of 5 CNNs.

**The method, in the words of the GAN slides.** The **generator** turns noise z plus a condition z̄ = (observation phases, amplitude, class) into a light curve that should "look real". The **discriminator** gets real and generated ("fake") curves with the same condition and has two outputs: a real-vs-fake score, and a class prediction (AC-GAN). The paper uses the Wasserstein loss with a gradient penalty (WGAN-GP) instead of the slides' cross-entropy, for more stable training. The discriminator is trained 5 times per generator step, and when one network trains the other one's weights are held fixed.

The paper's own contributions that we replicate:
- **γ-resampling**: training batches are drawn with class probabilities between the natural mix (γ = 0) and fully balanced (γ = 1). Balanced batches repeat the rare classes so often that the GAN over-fits them quickly.
- **G-score**: picks a checkpoint from *density* (fidelity) and *recall* (diversity) in a classifier's feature space. It is ~6× cheaper than training classifiers on every checkpoint, and it catches over-fitting that FID misses.

**What is not in the paper and had to be chosen.** Each choice is marked "ours" in the code:
- GAN batch size 64 and gradient-penalty weight λ = 10 (both WGAN-GP defaults);
- the classifier's batch size (128) and dropout (0.5), and how long it trains: it is checked on the validation set 4 times per epoch and the best check is kept (early stopping, patience 3 epochs). We measured that it over-fits after about 2 epochs: test accuracy falls from 74% to 67% by epoch 10;
- **mixed precision (fp16)** for the GAN and the classifiers. In full precision a T4 needs about 24 h for one GAN run, and the paper does not say what precision or hardware it used;
- k = 5 for the k-NN balls;
- amplitude = half the peak-to-peak range of the 100 kept points;
- the width of the class-range outlier filter: ±2σ, which reproduces the paper's Table 2 class counts (see the data section);
- the soft-warping interval for k is read as [1/(2b), 2/b] (the paper prints *a*, which breaks for magnitudes);
- for ZTF: bands, dates and period source are not stated in the paper; they were recovered from its plotted light curves (one curve per star and band in g and r, the label file's catalogue period, detections up to 19 July 2020). The paper's number of curves per class is then drawn at random from those that qualify (see the data section).

**How to run**
- Accelerator **GPU T4 ×2** · Internet **On** · Persistence: none needed. For `DATASET = "ztf"`, also *Add Input* → the dataset that holds `ztf_L40.npz`.
- First run with `SMOKE = True` interactively (about 20 minutes). It runs the tests and a tiny version of every job, and times the real jobs (in full and in mixed precision) so their hours can be estimated.
- Then set `SMOKE = False` → **Save Version → Save & Run All**. GPU 0 and GPU 1 each work down their own queue of jobs (see `QUEUES`). Every job saves a resumable state at least every 30 minutes and stops cleanly before the 10.5-hour budget.
- **Next session:** *Add Input* → this notebook's previous version (its output), set `SESSION` one higher, Save & Run All again. Finished jobs are skipped, an unfinished GAN resumes from its last state.
