"""The benchmark's search grids, the configurations chosen by the search, and the measured speeds.

This is the only file to edit between the stages (then rebuild the notebooks with build/build_notebooks.py):
after the search, put each family's winner into FINAL[family] and the speed its search used into SPEED[family].
The winner: the highest mean of the three best checkpoints' validation scores, read against the copy share
(a configuration whose good checkpoints copy its training curves loses; group decision, 2026-10-22).
Pure Python without torch, so the notebook builder can read it as well.

The search runs every configuration on split 0 and scores validation stars only. What is held fixed for all
families (the controls): network size (about 4 M parameters), EMA 0.999, resampling gamma = 0.25, mixed
precision, the schedule and the stopping rule (astra/bench_run.py). Sampling settings (steps, guidance, solver,
temperature) need no training and are swept at the selected checkpoint in the final stage.
"""
GRID = {
    # the paper's WGAN-GP (5 critic steps, gradient penalty 10, Adam betas 0.5 / 0.9): learning rate x batch size
    "gan": {f"lr{lr}_b{b}": dict(lr=float(lr), batch=b) for lr in ("5e-5", "1e-4", "2e-4") for b in (64, 256)},
    # KL weight beta (< 1: sharper curves, codes further from the prior; > 1: closer to the prior) x learning rate;
    # beta = 16 added for the search's session 2 (group decision, 2026-10-22): the score rose with beta up to the grid edge
    "vae": {f"b{b}_lr{lr}": dict(beta=float(b), lr=float(lr)) for b in ("0.1", "1", "4", "16") for lr in ("2e-4", "1e-3")},
    # what the network predicts (eps, or v = sqrt(ab) eps - sqrt(1 - ab) x0), cosine schedule x learning rate
    "ddpm": {f"{p}_lr{lr}": dict(pred=p, lr=float(lr)) for p in ("eps", "v") for lr in ("1e-4", "2e-4", "4e-4")},
    # how training times are drawn (uniform, or logit-normal: more at middle times, Esser et al. 2024) x learning rate
    "fm": {f"{t[:2]}_lr{lr}": dict(t_sampling=t, lr=float(lr)) for t in ("uniform", "logit_normal")
           for lr in ("1e-4", "2e-4", "4e-4")},
}

# the search's winner per family (a key of GRID[family]); None until the search is read
# (GAN, DDPM, FM: group decision, 2026-10-22; VAE: 2026-10-29, the rule and the copy-aware reading agree;
# notes/benchmark_results.md §5-§9)
FINAL = {"gan": "lr5e-5_b64", "vae": "b16_lr1e-3", "ddpm": "eps_lr1e-4", "fm": "un_lr4e-4"}

# training steps per second of each configuration ALONE on one T4 (fp16). They turn the schedule (minutes of
# training) into iterations, so checkpoints, caps and the compute budgets do not depend on how many jobs share a
# GPU. None here = the search notebook measures them in its first session and keeps them in runs_*/speeds.json;
# before the final stage they are copied here from the search output (the values that search used).
SPEED = {fam: {tag: None for tag in grid} for fam, grid in GRID.items()}
# GAN: runs_gan_catalina/speeds.json of the search (KM, 21 Oct)
SPEED["gan"].update({"lr1e-4_b256": 9.06, "lr1e-4_b64": 11.97, "lr2e-4_b256": 9.02, "lr2e-4_b64": 12.13,
                     "lr5e-5_b256": 8.96, "lr5e-5_b64": 11.68})
# DDPM, FM: the SMOKE speeds that were typed into the search notebooks (KT, 19 and 21 Oct)
SPEED["ddpm"].update({"eps_lr1e-4": 11.6, "eps_lr2e-4": 11.5, "eps_lr4e-4": 11.6, "v_lr1e-4": 11.5,
                      "v_lr2e-4": 11.5, "v_lr4e-4": 11.4})
SPEED["fm"].update({"un_lr1e-4": 11.7, "un_lr2e-4": 11.7, "un_lr4e-4": 11.7, "lo_lr1e-4": 11.7,
                    "lo_lr2e-4": 11.6, "lo_lr4e-4": 11.6})
# VAE: the winner's exact value from runs_vae_catalina/speeds.json of its search (part D, 25 Oct); the other
# configurations stay None (their speeds are in the search outputs)
SPEED["vae"].update({"b16_lr1e-3": 77.42449013496925})


def search_jobs(family):
    return [f"{family}_{tag}_s0" for tag in GRID[family]]


FINAL_SPLITS = (0, 1, 2, 3, 4)          # the final stage runs the winner on these splits (5 since 2026-10-15)


def final_jobs(family, splits=FINAL_SPLITS):
    """The winner on the given splits; every final run starts fresh in the final stage's own runs folder."""
    tag = FINAL[family]
    return [f"{family}_{tag}_s{s}" for s in splits] if tag else []
