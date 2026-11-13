"""Worker for the VAE notebook: trains the PELS-VAE models of a queue on one GPU, skipping finished ones.

python -m astra.run_vae --queue tcn,gru --deadline <unix time> [--smoke] [--runs-dir DIR]
"""
import argparse
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .common import WORK, Deadline, Logger, load_json, restore_rng, save_json, torch_save
from .pelsvae import load_raw, make_dataset, normalize_each, split_indices
from .tcn_vae import VAE_RNN, VAE_TCN, vae_loss

# The hyper-parameters of their best run (W&B run b68z1hwo, printed in their notebook). The paper's Table 2
# marks 7 blocks and kernel 7 in bold, but the model behind their figures used 9 blocks and kernel 5.
HP = dict(latent=4, hidden=48, levels=9, kernel=5, dropout=0.2, beta=0.75, lr=1e-3, batch=128, epochs=150,
          t_max=50, lr_min=1e-5, rnn_layers=2, split_seed=13)
JOBS = {
    "tcn": dict(arch="tcn", phy="", seed=13),              # cVAE: class-conditional (their best model)
    "tcn_phys": dict(arch="tcn", phy="PTA", seed=13),      # cVAE-P: + period, Teff, absolute G mag in the encoder
    "gru": dict(arch="gru", phy="", seed=13),              # the GRU curve of their Fig. 4
    "tcn_s14": dict(arch="tcn", phy="", seed=14),          # second seeds (same split, new init and batch order)
    "tcn_phys_s14": dict(arch="tcn", phy="PTA", seed=14),
}
SMOKE = dict(epochs=2, n_rows=3000)          # tiny jobs: a random 3,000-curve subset, 2 epochs


def cosine_lr(epoch, c):
    """Their CosineAnnealingLR(T_max=50, eta_min=1e-5) stepped as sch.step(epoch): the closed form, so over
    150 epochs the rate goes down, up and down again."""
    return c["lr_min"] + (c["lr"] - c["lr_min"]) * (1 + math.cos(math.pi * epoch / c["t_max"])) / 2


def build_model(c, n_classes, phy_dim):
    if c["arch"] == "tcn":
        return VAE_TCN(c["latent"], 600, c["kernel"], c["hidden"], c["levels"], 3, c["dropout"], n_classes, phy_dim)
    return VAE_RNN(c["latent"], 600, c["hidden"], c["rnn_layers"], 3, c["dropout"], n_classes, phy_dim)


def run_job(c, X, Y, P, tr, va, out, deadline, dev, log, n_classes=8):
    torch.manual_seed(c["seed"])
    onehot = torch.eye(n_classes, device=dev)
    model = build_model(c, n_classes, P.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    hist, start = [], 1
    if (out / "last.pt").exists():
        s = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        hist, start = s["hist"], s["epoch"] + 1
        restore_rng(s, dev)
        log(f"resumed after epoch {s['epoch']}")
    log(f"{c['arch']} phy='{c['phy']}': {sum(p.numel() for p in model.parameters()):,} parameters, "
        f"train {len(tr):,} / val {len(va):,}")

    def batches(idx, shuffle, epoch):
        if shuffle:
            g = torch.Generator().manual_seed(c["seed"] * 1000 + epoch)
            idx = idx[torch.randperm(len(idx), generator=g).numpy()]
        for i in range(0, len(idx), c["batch"]):
            j = torch.from_numpy(idx[i:i + c["batch"]]).to(dev)
            yield X[j], onehot[Y[j]], P[j]

    for epoch in range(start, c["epochs"] + 1):
        if deadline.passed(900):
            log(f"time budget reached before epoch {epoch}: resume next session")
            return "paused"
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = c["lr"] if epoch == 1 else cosine_lr(epoch - 1, c)
        model.train()
        sums = torch.zeros(3, device=dev); n = 0
        for x, lab, p in batches(tr, True, epoch):
            xhat, mu, logvar, _ = model(x, lab, p)
            loss, mse, kld = vae_loss(x, xhat, mu, logvar, c["beta"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sums += torch.stack([loss.detach(), mse.detach(), kld.detach()]); n += 1
        tr_l = (sums / n).tolist()
        model.eval()
        sums = torch.zeros(3, device=dev); n = 0
        with torch.no_grad():                       # like their _test_epoch: eval mode, z still sampled
            for x, lab, p in batches(va, False, epoch):
                xhat, mu, logvar, _ = model(x, lab, p)
                sums += torch.stack(vae_loss(x, xhat, mu, logvar, c["beta"])); n += 1
        va_l = (sums / n).tolist()
        hist.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": tr_l[0], "train_mse": tr_l[1],
                     "train_kl": tr_l[2], "val_loss": va_l[0], "val_mse": va_l[1], "val_kl": va_l[2],
                     "seconds": time.time() - t0})
        torch_save({"model": model.state_dict(), "opt": opt.state_dict(), "hist": hist, "epoch": epoch,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if dev == "cuda" else None}, out / "last.pt")
        save_json(hist, out / "history.json")
        if epoch == 1 or epoch % 10 == 0 or epoch == c["epochs"]:
            log(f"epoch {epoch}: train loss {tr_l[0]:.5f} (mse {tr_l[1]:.5f}, kl {tr_l[2]:.5f}) | "
                f"val loss {va_l[0]:.5f} | lr {hist[-1]['lr']:.2e} | {hist[-1]['seconds']:.0f} s")
    torch_save(model.state_dict(), out / "model.pt")
    # latent means and std of every curve (dataset order), used for Fig. 6 and the regressions of Table 3
    mus, stds = [], []
    with torch.no_grad():
        allidx = np.arange(len(Y))
        for x, lab, p in batches(allidx, False, 0):
            mu, logvar = model.encoder(x, lab, p)
            mus.append(mu.cpu()); stds.append(torch.exp(0.5 * logvar).cpu())
        # reconstructions from the latent mean for 3 validation curves per class (their Figs. 5 and 7)
        rng = np.random.default_rng(0)
        yva = Y[torch.from_numpy(va).to(dev)].cpu().numpy()
        show = np.concatenate([rng.choice(va[yva == k], min(3, int(np.sum(yva == k))), replace=False)
                               for k in range(n_classes)])
        j = torch.from_numpy(show).to(dev)
        mu, _ = model.encoder(X[j], onehot[Y[j]], P[j])
        rec = model.decoder(mu, X[j][:, :, 0], onehot[Y[j]])
    np.savez(out / "latent.npz", mu=torch.cat(mus).numpy(), std=torch.cat(stds).numpy())
    np.savez(out / "recon.npz", idx=show, x=X[j].cpu().numpy(), xhat=rec[:, :, 0].cpu().numpy())
    (out / "last.pt").unlink(missing_ok=True)        # keep only the final weights
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--deadline", type=float, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs-dir", default=str(WORK / "runs"))
    a = ap.parse_args()
    runs_dir = Path(a.runs_dir)
    tiny = a.smoke or os.environ.get("ASTRA_LOCAL_TEST") == "1"      # my offline CPU test runs tiny jobs
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    deadline = Deadline(a.deadline)
    torch.backends.cudnn.benchmark = True
    lcs, meta = load_raw(WORK / "data")
    lcs = normalize_each(lcs)
    for job in a.queue.split(","):
        status_path = runs_dir / job / "status.json"
        if (load_json(status_path, {}) or {}).get("state") == "done":
            print(f"[{job}] already done, skipping", flush=True)
            continue
        out = runs_dir / job
        log = Logger(out / "log.txt")
        c = dict(HP, **JOBS[job], **(SMOKE if tiny else {}))
        save_json(c, out / "config.json")
        log(f"start {job} ({'tiny' if tiny else 'real run'})")
        t0 = time.time()
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            ds = make_dataset(lcs, meta, c["phy"])
            if tiny:
                sub = np.sort(np.random.default_rng(0).choice(len(ds["y"]), c["n_rows"], replace=False))
                ds = dict(ds, **{k: ds[k][sub] for k in ("x", "y", "phy", "rows")})
            tr, va = split_indices(len(ds["y"]), c["split_seed"])
            np.savez(out / "split.npz", rows=ds["rows"], train=tr, val=va)
            X = torch.from_numpy(ds["x"]).to(dev)
            Y = torch.from_numpy(ds["y"]).to(dev)
            P = torch.from_numpy(ds["phy"]).to(dev)
            state = run_job(c, X, Y, P, tr, va, out, deadline, dev, log, len(ds["classes"]))
            del X, Y, P
        except Exception:
            log(traceback.format_exc())
            state = "failed"
        prev = load_json(status_path, {}) or {}
        save_json({"state": state, "hours_this_session": (time.time() - t0) / 3600,
                   "hours_total": prev.get("hours_total", 0) + (time.time() - t0) / 3600,
                   "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0}, status_path)
        log(f"{job}: {state}")
        if state == "paused":
            break


if __name__ == "__main__":
    main()
