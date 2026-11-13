"""Worker for the GAN notebook: runs a queue of jobs on one GPU, skipping finished ones.

python -m astra.run_gan --queue trtr,gan_u_s0 --deadline <unix time> [--smoke]
"""
import argparse
import copy
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .classifier import CNNEnsemble, fit_ensembles
from .common import WORK, Deadline, Logger, load_json, save_json
from .gan import GanRun, load_data, subset
from .warping import make_warp

# Settings from the paper (Sec. 4.3-4.4); the ones it does not state are marked "ours".
GAN = dict(latent=16, iters=400_000, n_disc=5, lr=1e-4, xi=1e-3, eps_drift=1.0, ema=0.999, ckpt_every=10_000,
           batch=64,            # ours: batch size is not given in the paper (64 as in WGAN-GP)
           amp=True,            # ours: fp16 mixed precision on the GPU (fp32 needs ~24 h per run on a T4)
           lambda_gp=10.0,      # ours: GP weight not given (10 as in WGAN-GP)
           k=5, dr_reps=5,      # ours: k for the k-NN balls (Naeem et al. default); 5 repetitions as in Fig. 7
           fid_batches=10, fid_batch_size=5000,
           sel_runs=3, ens=5, sel_max_epochs=6, sel_patience=2,        # val TSTR per checkpoint (ours: 3 runs)
           final_runs=15, final_max_epochs=10, final_patience=3,       # Table 5: 15 runs of 5-CNN ensembles
           # (ours: epochs and patience; measured: the classifier's validation loss bottoms out after ~2 epochs)
           save_every_s=1800)
JOBS = {
    "gan_gamma025_s0": dict(gamma=0.25, seed=0),     # gamma-GAN (rows G, H of Table 5)
    "gan_u_s0": dict(gamma=1.0, seed=0),             # u-GAN (rows E, F)
    "gan_gamma025_s1": dict(gamma=0.25, seed=1),     # extra seeds of the gamma-GAN
    "gan_gamma025_s2": dict(gamma=0.25, seed=2),
    "gan_gamma0_s0": dict(gamma=0.0, seed=0),        # no resampling (the gamma = 0 curve of Fig. 4)
}
TRTR_ROWS = {"A_oversampling": None, "B_soft_time_warping": "time", "C_soft_mag_warping": "mag",
             "D_soft_mixed_warping": "mixed"}
SMOKE = dict(iters=300, ckpt_every=100, dr_reps=1, fid_batches=2, fid_batch_size=500, sel_runs=1,
             sel_max_epochs=2, sel_patience=1, final_runs=2, final_max_epochs=2, final_patience=1, save_every_s=60)


def run_trtr(data, runs_dir, deadline, smoke, log, dev):
    """Rows A-D of Table 5: classifiers trained on real data (balanced by oversampling, +- soft warping)."""
    out = runs_dir / "trtr" / "trtr.json"
    res = load_json(out, {})
    kw = dict(runs=2, ens=5, max_epochs=2, patience=1) if smoke else \
        dict(runs=GAN["final_runs"], ens=GAN["ens"], max_epochs=GAN["final_max_epochs"], patience=GAN["final_patience"])
    tr, va, te = (subset(data, data[f"{s}_idx"]) for s in ("train", "val", "test"))
    for row, mode in TRTR_ROWS.items():
        if row in res:
            continue
        if deadline.passed(1800):
            log("time budget reached; the remaining TRTR rows run next session")
            return "paused"
        t0 = time.time()
        r = fit_ensembles(tr, va, te, data["n_classes"], warp=make_warp(mode), seed=0, device=dev, log=log,
                          tag=row, **kw)
        r["minutes"] = (time.time() - t0) / 60
        res[row] = r
        save_json(res, out)
        log(f"{row}: test accuracy {100 * np.mean(r['test_acc']):.2f} +- {100 * np.std(r['test_acc']):.2f} "
            f"({r['minutes']:.1f} min)")
    return "done"


def benchmark(data, feat, dev, tmp_dir, latent=16):
    """SMOKE only: times the real-size GAN step and classifier epochs, to estimate every job's hours."""
    out = {}
    for amp in (False, True):           # full precision first, then mixed precision (the one the jobs use)
        run = GanRun(dict(GAN, gamma=0.25, seed=0, amp=amp, latent=latent), data, feat, tmp_dir,
                     Deadline(time.time() + 1e6), dev)
        for _ in range(20):
            run.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(60):
            last = run.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        out["gan_it_per_s" if amp else "gan_it_per_s_fp32"] = 60 / (time.time() - t)
    out["amp_skipped_D_steps"] = float(run.skipped)
    out["amp_scale_D"] = run.scalerD.get_scale() if run.amp else 0.0
    out["amp_last_losses_D_G_W"] = [round(v, 3) for v in last.tolist()]
    t = time.time()
    run.cfg = dict(run.cfg, sel_max_epochs=1)
    run.eval_checkpoint(copy.deepcopy(run.G_ema), 0)
    out["ckpt_eval_1epoch_s"] = time.time() - t
    tr, va, te = (subset(data, data[f"{s}_idx"]) for s in ("train", "val", "test"))
    for runs in (GAN["sel_runs"], GAN["final_runs"]):
        t = time.time()
        r = fit_ensembles(tr, va, te, data["n_classes"], runs=runs, ens=GAN["ens"], max_epochs=1, device=dev,
                          log=lambda *a: None)
        out[f"clf_epoch_s_{runs * GAN['ens']}cnn"] = time.time() - t
    out["clf_1epoch_test_acc"] = float(np.mean(r["test_acc"]))
    t = time.time()
    fit_ensembles(tr, va, te, data["n_classes"], runs=GAN["sel_runs"], ens=GAN["ens"], max_epochs=1, device=dev,
                  log=lambda *a: None, amp=False)
    out[f"clf_epoch_s_{GAN['sel_runs'] * GAN['ens']}cnn_fp32"] = time.time() - t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--deadline", type=float, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs-dir", default=str(WORK / "runs"))
    ap.add_argument("--data", default=str(WORK / "data" / "catalina_L100.npz"))
    ap.add_argument("--latent", type=int, default=GAN["latent"])       # 16 for Catalina, 8 for ZTF (Sec. 4.2)
    a = ap.parse_args()
    runs_dir = Path(a.runs_dir)
    tiny = a.smoke or os.environ.get("ASTRA_LOCAL_TEST") == "1"     # my offline CPU test runs tiny jobs
    deadline = Deadline(a.deadline)
    dev = "cuda" if torch.cuda.is_available() else "cpu"      # cpu only in my offline tests
    torch.backends.cudnn.benchmark = True
    data = load_data(a.data, dev)
    feat = CNNEnsemble(1, data["n_classes"], data["x"].shape[1]).to(dev)
    feat.load_state_dict(torch.load(runs_dir / "feature_cnn.pt", map_location=dev))
    feat.eval()
    for job in a.queue.split(","):
        status_path = runs_dir / job / "status.json"
        if (load_json(status_path, {}) or {}).get("state") == "done":
            print(f"[{job}] already done, skipping", flush=True)
            continue
        log = Logger(runs_dir / job / "log.txt")
        log(f"start {job} ({'tiny' if tiny else 'real run'})")
        t0 = time.time()
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            if job == "trtr":
                state = run_trtr(data, runs_dir, deadline, tiny, log, dev)
            else:
                cfg = dict(GAN, **JOBS[job], **(SMOKE if tiny else {}), latent=a.latent)
                save_json(cfg, runs_dir / job / "config.json")
                state = GanRun(cfg, data, feat, runs_dir / job, deadline, dev).run(reserve_s=600)
        except Exception:
            log(traceback.format_exc())
            state = "failed"
        prev = load_json(status_path, {}) or {}
        save_json({"state": state, "hours_this_session": (time.time() - t0) / 3600,
                   "hours_total": prev.get("hours_total", 0) + (time.time() - t0) / 3600,
                   "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0}, status_path)
        log(f"{job}: {state}")
        if state == "paused":
            break                 # out of time: the rest of the queue waits for the next session


if __name__ == "__main__":
    main()
