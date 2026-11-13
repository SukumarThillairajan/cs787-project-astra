"""The conditional WGAN-GP of Garcia-Jara et al. (2022), Sec. 4.2-4.7 and Table 4.

Generator G(z, zbar) with zbar = [phases (L), amplitude (1), one-hot class (N)]: Dense+ReLU -> 4 x 1024,
five transposed-conv blocks (stride 2, kernel 5) 1024->512->256->128->64->1, output tanh * s.
Discriminator D(x, zbar-tensor): five conv blocks (stride 2, kernel 5, LeakyReLU) 64->128->256->512->1024,
Dense -> N + 1 outputs: D_rg (real vs fake score) and D_y (class logits, the AC-GAN part).

Losses (Eqs. 1, 2, 10, 11), minimised:
  L_D = E[D_rg(x_g)] - E[D_rg(x_r)] + lambda * GP + xi * (H_r + H_g) + eps * E[D_rg(x_r)^2]
  L_G = -E[D_rg(x_g)] + xi * H_g
with n_disc = 5 D steps per G step, Adam(1e-4, 0.5, 0.9), 400k G iterations, EMA(0.999) of G's weights.

Training uses mixed precision (fp16 on the GPU, cfg["amp"]): in full precision a T4 does ~4.6 iterations/s,
i.e. 24 h for one run. The gradient penalty follows PyTorch's AMP recipe (scale the output before
autograd.grad, un-scale the gradient). Generation and all metrics stay in full precision.
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import amp_on, autocast, extract_features, fit_ensembles
from .common import Logger, load_json, restore_rng, save_json, torch_save
from .metrics import fid_from_stats, g_score, gauss_stats, knn_radii, prdc
from .sampling import ClassResampler

TRAIN, VAL, TEST = 0, 1, 2
GP_SCALE_MAX = 1024.0


class Generator(nn.Module):
    def __init__(self, latent, L, n_classes, scale):
        super().__init__()
        self.L, self.n_classes, self.latent = L, n_classes, latent
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.fc = nn.Linear(latent + L + 1 + n_classes, 4 * 1024)
        chans = [1024, 512, 256, 128, 64, 1]
        self.deconv = nn.ModuleList(
            [nn.ConvTranspose1d(cin, cout, 5, stride=2, padding=2, output_padding=1)
             for cin, cout in zip(chans[:-1], chans[1:])])

    def forward(self, z, ph, amp, onehot):
        h = F.relu(self.fc(torch.cat([z, ph, amp[:, None], onehot], 1))).view(-1, 1024, 4)
        for i, layer in enumerate(self.deconv):
            h = layer(h)
            if i < len(self.deconv) - 1:
                h = F.relu(h)
        return (torch.tanh(h) * self.scale)[:, 0, :self.L]     # length 4 * 2^5 = 128, keep the first L


class Discriminator(nn.Module):
    def __init__(self, L, n_classes):
        super().__init__()
        chans = [1 + 2 + n_classes, 64, 128, 256, 512, 1024]
        self.conv = nn.ModuleList([nn.Conv1d(cin, cout, 5, stride=2, padding=2)
                                   for cin, cout in zip(chans[:-1], chans[1:])])
        l = L
        for _ in range(5):
            l = math.ceil(l / 2)
        self.fc = nn.Linear(1024 * l, n_classes + 1)

    def forward(self, x, ph, amp, onehot):
        L = x.shape[1]
        cond = torch.cat([ph[:, None], amp[:, None, None].expand(-1, 1, L), onehot[:, :, None].expand(-1, -1, L)], 1)
        h = torch.cat([x[:, None], cond], 1)
        for layer in self.conv:
            h = F.leaky_relu(layer(h), 0.2)
        out = self.fc(h.flatten(1))
        return out[:, 0], out[:, 1:]                           # D_rg, D_y logits


def gradient_penalty(D, x_r, x_g, cond, scaler=None, amp=False):
    eps = torch.rand(len(x_r), 1, device=x_r.device)
    x_hat = (eps * x_r + (1 - eps) * x_g).requires_grad_(True)
    with autocast(x_r.device.type, amp):
        d_hat, _ = D(x_hat, *cond)
    # AMP recipe: scale the output so small gradients do not underflow in fp16, then divide the scale out again.
    # The scale is capped: the scaler starts at 65,536, and that already overflows fp16 (max 65,504) here,
    # because the gradient of the sum with respect to every critic output is exactly the scale.
    scale = min(scaler.get_scale(), GP_SCALE_MAX) if scaler is not None and scaler.is_enabled() else 1.0
    grad, = torch.autograd.grad(d_hat.float().sum() * scale, x_hat, create_graph=True)
    return (((grad / scale).norm(2, dim=1) - 1) ** 2).mean()


@torch.no_grad()
def generate(G, data, idx, seed, batch=4096):
    """One synthetic curve per index: the conditions (phases, amplitude, class) of real training star idx[i]."""
    g = torch.Generator(device=data["x"].device).manual_seed(seed)
    out = []
    for i in range(0, len(idx), batch):
        j = idx[i:i + batch]
        z = torch.randn(len(j), G.latent, device=data["x"].device, generator=g)
        out.append(G(z, data["ph"][j], data["amp"][j], data["onehot"][j]))
    return torch.cat(out)


def balanced_idx(data, n, seed, gamma=1.0):
    """n training-set indices from the resampling block (gamma = 1: balanced, i.e. D_train^u)."""
    rs = ClassResampler(data["y_np"][data["train_idx_np"]], gamma, seed=seed, n_classes=data["n_classes"])
    return data["train_idx"][torch.from_numpy(rs.sample(n)).to(data["x"].device)]


def load_data(npz_path, device):
    z = np.load(npz_path)
    n_classes = int(z["y"].max()) + 1
    d = {k: torch.from_numpy(z[k]).to(device) for k in ("phase", "mag", "amp", "y")}
    data = {"x": d["mag"].float(), "ph": d["phase"].float(), "amp": d["amp"].float(), "y": d["y"].long(),
            "onehot": F.one_hot(d["y"].long(), n_classes).float(), "n_classes": n_classes,
            "y_np": z["y"], "split": z["split"]}
    for name, s in (("train", TRAIN), ("val", VAL), ("test", TEST)):
        idx = np.flatnonzero(z["split"] == s)
        data[f"{name}_idx_np"] = idx
        data[f"{name}_idx"] = torch.from_numpy(idx).to(device)
    data["scale"] = float(d["mag"][data["train_idx"]].abs().max())
    return data


def subset(data, idx, x=None):
    """A split as the classifier wants it: x, ph, y (optionally with synthetic magnitudes x)."""
    return {"x": data["x"][idx] if x is None else x, "ph": data["ph"][idx], "y": data["y"][idx]}


class GanRun:
    """One GAN training run with checkpoints, per-checkpoint evaluation, resume and final test scores."""

    def __init__(self, cfg, data, feat_model, out_dir, deadline, device="cuda"):
        self.cfg, self.data, self.feat, self.dev = cfg, data, feat_model, device
        self.out, self.deadline = out_dir, deadline
        (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        self.log = Logger(out_dir / "log.txt")
        self.metrics_path = out_dir / "metrics.json"
        self.metrics = load_json(self.metrics_path, [])
        torch.manual_seed(cfg["seed"])
        N, L = data["n_classes"], data["x"].shape[1]
        self.G = Generator(cfg["latent"], L, N, data["scale"]).to(device)
        self.D = Discriminator(L, N).to(device)
        self.G_ema = copy.deepcopy(self.G).eval()
        self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg["lr"], betas=(0.5, 0.9))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg["lr"], betas=(0.5, 0.9))
        self.amp = amp_on(cfg.get("amp", False), device)
        self.scalerG = torch.amp.GradScaler(enabled=self.amp)
        self.scalerD = torch.amp.GradScaler(enabled=self.amp)
        self.skipped = 0.0                 # D steps with a non-finite loss (AMP skips those updates)
        ytr = data["y_np"][data["train_idx_np"]]
        self.sampler = ClassResampler(ytr, cfg["gamma"], seed=cfg["seed"], n_classes=N)
        self.it = 0
        self._real_cache = None

    # ---------------- state ----------------
    def save_state(self):
        torch_save({"G": self.G.state_dict(), "D": self.D.state_dict(), "G_ema": self.G_ema.state_dict(),
                    "optG": self.optG.state_dict(), "optD": self.optD.state_dict(), "sampler": self.sampler,
                    "scalerG": self.scalerG.state_dict(), "scalerD": self.scalerD.state_dict(),
                    "it": self.it, "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if self.dev == "cuda" else None}, self.out / "last.pt")

    def load_state(self):
        p = self.out / "last.pt"
        if not p.exists():
            return False
        s = torch.load(p, map_location=self.dev, weights_only=False)
        self.G.load_state_dict(s["G"]); self.D.load_state_dict(s["D"]); self.G_ema.load_state_dict(s["G_ema"])
        self.optG.load_state_dict(s["optG"]); self.optD.load_state_dict(s["optD"])
        self.scalerG.load_state_dict(s["scalerG"]); self.scalerD.load_state_dict(s["scalerD"])
        self.sampler, self.it = s["sampler"], s["it"]
        restore_rng(s, self.dev)
        return True

    # ---------------- training ----------------
    def batch(self, idx_np):
        d = self.data
        j = d["train_idx"][torch.from_numpy(idx_np).to(self.dev)]
        return d["x"][j], (d["ph"][j], d["amp"][j], d["onehot"][j]), d["y"][j]

    def step(self):
        c, B = self.cfg, self.cfg["batch"]
        for _ in range(c["n_disc"]):
            x_r, cond, y = self.batch(self.sampler.sample(B))
            with torch.no_grad(), autocast(self.dev, self.amp):
                x_g = self.G(torch.randn(B, c["latent"], device=self.dev), *cond).float()
            with autocast(self.dev, self.amp):
                d_r, logit_r = self.D(x_r, *cond)
                d_g, logit_g = self.D(x_g, *cond)
            d_r, d_g = d_r.float(), d_g.float()
            gp = gradient_penalty(self.D, x_r, x_g, cond, self.scalerD, self.amp)
            h = F.cross_entropy(logit_r.float(), y) + F.cross_entropy(logit_g.float(), y)
            loss_d = d_g.mean() - d_r.mean() + c["lambda_gp"] * gp + c["xi"] * h + c["eps_drift"] * (d_r ** 2).mean()
            self.optD.zero_grad(set_to_none=True)
            self.scalerD.scale(loss_d).backward()
            self.scalerD.step(self.optD)
            self.scalerD.update()
            self.skipped = self.skipped + (~torch.isfinite(loss_d.detach())).float()
        _, cond, y = self.batch(self.sampler.sample(B))
        with autocast(self.dev, self.amp):
            x_g = self.G(torch.randn(B, c["latent"], device=self.dev), *cond)
            d_g, logit_g = self.D(x_g, *cond)
        d_g = d_g.float()
        loss_g = -d_g.mean() + c["xi"] * F.cross_entropy(logit_g.float(), y)
        self.optG.zero_grad(set_to_none=True)
        self.scalerG.scale(loss_g).backward()
        self.scalerG.step(self.optG)
        self.scalerG.update()
        with torch.no_grad():
            for pe, p in zip(self.G_ema.parameters(), self.G.parameters()):
                pe.lerp_(p, 1 - c["ema"])
        self.it += 1
        return torch.nan_to_num(torch.stack([loss_d.detach(), loss_g.detach(), (d_r.mean() - d_g.mean()).detach()]))

    # ---------------- evaluation of one checkpoint ----------------
    def real_cache(self):
        """Features of the real training set, per class, with their k-NN radii (computed once)."""
        if self._real_cache is None:
            d = self.data
            tr = d["train_idx"]
            f = extract_features(self.feat, d["x"][tr], d["ph"][tr])
            ytr = d["y"][tr]
            per = []
            for c in range(d["n_classes"]):
                fc = f[ytr == c]
                per.append((fc, knn_radii(fc, self.cfg["k"])))
            self._real_cache = (gauss_stats(f.cpu().numpy().astype(np.float64)), per)
        return self._real_cache

    def eval_checkpoint(self, G, it):
        c, d = self.cfg, self.data
        t0 = time.time()
        real_stats, per = self.real_cache()
        n_gen = len(d["train_idx"])
        # density & recall per class, repeated on fresh generated sets (mean over repetitions)
        reps = []
        for r in range(c["dr_reps"]):
            idx = balanced_idx(d, n_gen, seed=10_000 + r)
            fg = extract_features(self.feat, generate(G, d, idx, seed=20_000 + r), d["ph"][idx])
            yg = d["y"][idx]
            reps.append([prdc(per[k][0], fg[yg == k], c["k"], real_radii=per[k][1]) for k in range(d["n_classes"])])
        res = {"it": it}
        for m in ("density", "recall", "precision", "coverage"):
            arr = np.array([[cls[m] for cls in rep] for rep in reps])       # (reps, classes)
            res[m] = arr.mean(0).tolist()
            res[m + "_std"] = arr.std(0).tolist()
        # FID: 10 batches of 5000 generated curves (conditions at the natural class mix) vs the real train set
        fids = []
        for b in range(c["fid_batches"]):
            idx = balanced_idx(d, c["fid_batch_size"], seed=30_000 + b, gamma=0.0)
            fg = extract_features(self.feat, generate(G, d, idx, seed=40_000 + b), d["ph"][idx])
            fids.append(fid_from_stats(*real_stats, *gauss_stats(fg.cpu().numpy().astype(np.float64))))
        res["fid"] = float(np.mean(fids))
        # validation TSTR accuracy: classifiers trained on a balanced generated set, scored on real val
        idx = balanced_idx(d, n_gen, seed=50_000 + it)
        syn = subset(d, idx, x=generate(G, d, idx, seed=60_000 + it))
        r = fit_ensembles(syn, subset(d, d["val_idx"]), subset(d, d["test_idx"]), d["n_classes"],
                          runs=c["sel_runs"], ens=c["ens"], max_epochs=c["sel_max_epochs"],
                          patience=c["sel_patience"], seed=it, device=self.dev, log=lambda *a: None,
                          amp=c.get("amp", False))
        res["val_acc_runs"] = r["val_acc"]
        res["val_acc"] = float(np.mean(r["val_acc"]))
        res["test_acc_info"] = float(np.mean(r["test_acc"]))       # logged only, never used for selection
        res["eval_s"] = time.time() - t0
        return res

    def evaluate_pending(self):
        done = {m["it"] for m in self.metrics}
        for p in sorted((self.out / "ckpt").glob("G_ema_*.pt")):
            it = int(p.stem.split("_")[-1])
            if it in done:
                continue
            G = copy.deepcopy(self.G_ema)
            G.load_state_dict(torch.load(p, map_location=self.dev))
            m = self.eval_checkpoint(G.eval(), it)
            self.metrics.append(m)
            self.metrics.sort(key=lambda a: a["it"])
            save_json(self.metrics, self.metrics_path)
            self.log(f"checkpoint {it:,}: FID {m['fid']:.2f} | macro D {np.mean(m['density']):.3f} "
                     f"R {np.mean(m['recall']):.3f} | val TSTR acc {m['val_acc']:.3f} ({m['eval_s']:.0f} s)")

    # ---------------- the whole run ----------------
    def run(self, reserve_s):
        c = self.cfg
        if (self.out / "final.json").exists():
            self.log("already finished")
            return "done"
        if self.load_state():
            self.log(f"resumed at iteration {self.it:,}")
        self.evaluate_pending()
        t_save, t0, it0 = time.time(), time.time(), self.it
        hist, n_hist = 0.0, 0         # running sums stay on the GPU (no sync every step)
        while self.it < c["iters"]:
            if self.deadline.passed(reserve_s):
                self.save_state()
                self.log(f"time budget reached at iteration {self.it:,}: state saved, resume next session")
                return "paused"
            hist = hist + self.step()
            n_hist += 1
            if self.it % c["ckpt_every"] == 0:
                torch_save(self.G_ema.state_dict(), self.out / "ckpt" / f"G_ema_{self.it:07d}.pt")
                self.save_state()
                t_save = time.time()
                h = (hist / max(n_hist, 1)).tolist()
                speed = (self.it - it0) / (time.time() - t0)
                self.log(f"iter {self.it:,}: loss D {h[0]:.3f} G {h[1]:.3f} | W-dist est. {h[2]:.3f} | "
                         f"{speed:.1f} it/s" + (f" | amp scale {self.scalerD.get_scale():.0f}, skipped D steps "
                                                f"{float(self.skipped):.0f}" if self.amp else ""))
                with open(self.out / "train_curve.csv", "a") as f:
                    f.write(f"{self.it},{h[0]:.5f},{h[1]:.5f},{h[2]:.5f}\n")
                hist, n_hist = 0.0, 0
                self.evaluate_pending()
                t0, it0 = time.time(), self.it
            elif time.time() - t_save > c["save_every_s"]:
                self.save_state()
                t_save = time.time()
        self.save_state()
        self.evaluate_pending()
        return self.finish(reserve_s)

    def finish(self, reserve_s):
        """Model selection (val TSTR accuracy and G-score) and the final TSTR test scores (15 runs x 5 CNNs)."""
        c, d = self.cfg, self.data
        its = [m["it"] for m in self.metrics]
        G_sc = g_score([m["density"] for m in self.metrics], [m["recall"] for m in self.metrics])
        for m, g in zip(self.metrics, G_sc):
            m["g_score"] = float(g)
        save_json(self.metrics, self.metrics_path)
        choice = {"val_acc": its[int(np.argmax([m["val_acc"] for m in self.metrics]))],
                  "g_score": its[int(np.argmax(G_sc))]}
        final = load_json(self.out / "final_partial.json", {})
        for crit, it in choice.items():
            if crit in final:
                continue
            if self.deadline.passed(reserve_s / 2):
                self.log("time budget reached before the final evaluation; it resumes next session")
                return "paused"
            G = copy.deepcopy(self.G_ema)
            G.load_state_dict(torch.load(self.out / "ckpt" / f"G_ema_{it:07d}.pt", map_location=self.dev))
            idx = balanced_idx(d, len(d["train_idx"]), seed=70_000)
            syn = subset(d, idx, x=generate(G.eval(), d, idx, seed=80_000))
            r = fit_ensembles(syn, subset(d, d["val_idx"]), subset(d, d["test_idx"]), d["n_classes"],
                              runs=c["final_runs"], ens=c["ens"], max_epochs=c["final_max_epochs"],
                              patience=c["final_patience"], seed=1, device=self.dev, log=self.log, tag=crit,
                              amp=c.get("amp", False))
            r["checkpoint"] = it
            final[crit] = r
            save_json(final, self.out / "final_partial.json")
            self.log(f"{crit}: checkpoint {it:,} -> TSTR test accuracy "
                     f"{100 * np.mean(r['test_acc']):.2f} +- {100 * np.std(r['test_acc']):.2f}")
            # a few generated curves per class from this checkpoint, for the figures
            show = balanced_idx(d, 16 * d["n_classes"], seed=90_000)
            np.savez(self.out / f"samples_{crit}.npz", idx=show.cpu().numpy(),
                     x=generate(G, d, show, seed=91_000).cpu().numpy())
        save_json(final, self.out / "final.json")
        self.log("finished")
        return "done"
