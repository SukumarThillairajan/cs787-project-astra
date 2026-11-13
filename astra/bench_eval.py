"""The evaluation harness of the ASTRA benchmark. It is the same code for every model family: a family only
hands over sample(idx, seed) -> magnitudes of one generated curve per condition idx (rows of real stars).

What is measured, and where it comes from:
  - TSTR (train on synthetic, test on real) and real + synthetic training, with the classifier of Garcia-Jara
    et al. (2022): 15 runs of an ensemble of 5 CNNs. Reported on the real test stars of the split.
  - Balanced accuracy = the mean of the per-class recalls. On the paper's class-balanced test set this is its
    plain accuracy; on our stratified test set it keeps every class equally important. Macro-F1, accuracy and
    the per-class recalls are reported next to it.
  - Fidelity in the feature space of a classifier trained on real data (the paper's Sec. 4.3): per-class
    precision, recall, density and coverage with k-NN balls, and FID.
  - Memorisation (ours), two numbers per class, from the distance between a generated curve and the real curve
    of its own condition star. The ratio: median distance for training stars / the same for validation stars;
    near 1, training stars are treated like new stars, far below 1, the model reproduces the training curves it
    was conditioned on. The copy share: the share of training-star curves that are closer to their source than
    95% of the validation-star curves are to theirs; 5% means no copying, 100% means every curve is a copy.
    Notebook 01 showed why it is needed: late GAN checkpoints return the training curve of the condition for the
    rare classes, which density and recall cannot see.
  - TSTR for conditions no training star has (ours, 2026-10-22): one star's phases with another star's
    amplitude, same class; at the selected and at the last checkpoint. It tells whether a model that copies its
    training curves has still learnt the classes.
Model selection uses validation stars only. Test scores of the non-selected settings are saved for the
analysis but never used for a choice.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .bench_data import balanced_idx, subset
from .classifier import CNNEnsemble, amp_on, autocast, extract_features, predict_proba, stack_input
from .common import load_json, save_json
from .metrics import fid_from_stats, knn_radii, prdc
from .sampling import ClassResampler
from .warping import make_warp

# Classifier settings are those of the replication notebook (the "ours" choices are explained there).
EVAL = dict(k=5, ens=5,
            n_sel=20_000, sel_runs=3, sel_max_epochs=6, sel_patience=2,      # per checkpoint: 3 runs x 5 CNNs
            final_runs=15, final_max_epochs=10, final_patience=3,            # test scores: 15 runs x 5 CNNs
            p_syn=(0.25, 0.5, 0.75),      # share of synthetic curves in real + synthetic training
            n_final=None,                 # synthetic curves for the test scores (None = one per training curve)
            max_sweeps=None,              # how many of the family's other sampling settings to score (None = all)
            n_mem=2_000, n_loss=2_000)
EVAL_TINY = dict(n_sel=600, sel_runs=1, sel_max_epochs=1, sel_patience=1, final_runs=2, final_max_epochs=1,
                 final_patience=1, p_syn=(0.5,), n_final=1200, max_sweeps=2, n_mem=200, n_loss=200)
REAL_ROWS = {"A_oversampling": None, "B_soft_time_warping": "time", "C_soft_mag_warping": "mag",
             "D_soft_mixed_warping": "mixed"}


def save_npz(path, compressed=False, **arrays):
    """Write to a temp file first, so a kill mid-write never leaves a broken file."""
    tmp = path.with_name(path.stem + ".tmp.npz")
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------- scores
def confusion(y, pred, n_classes, w=None):
    """pred: (runs, n) predicted classes -> (runs, N, N) counts [true, predicted]; w = optional curve weights."""
    pred = np.atleast_2d(pred)
    return np.stack([np.bincount(y * n_classes + p, weights=w, minlength=n_classes ** 2)
                     .reshape(n_classes, n_classes) for p in pred])


def scores_from_conf(conf):
    """conf[..., true, predicted] -> balanced accuracy (mean recall over the classes present), macro-F1,
    accuracy, and the per-class recall, precision and F1."""
    conf = np.asarray(conf, float)
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    n_true, n_pred = conf.sum(-1), conf.sum(-2)
    rec, prec = tp / np.maximum(n_true, 1e-12), tp / np.maximum(n_pred, 1e-12)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    has = n_true > 0
    k = np.maximum(has.sum(-1), 1)
    return {"bal_acc": (rec * has).sum(-1) / k, "macro_f1": (f1 * has).sum(-1) / k,
            "acc": tp.sum(-1) / np.maximum(conf.sum((-2, -1)), 1e-12), "recall": rec, "precision": prec, "f1": f1}


def summarise(y, pred, n_classes, rare=None):
    """Scores of every run (pred: runs x n), then their mean and std over the runs."""
    s = scores_from_conf(confusion(y, pred, n_classes))
    out = {}
    for k in ("bal_acc", "macro_f1", "acc"):
        out[k], out[k + "_std"], out[k + "_runs"] = float(s[k].mean()), float(s[k].std()), s[k].tolist()
    for k in ("recall", "precision", "f1"):
        out[k] = s[k].mean(0).tolist()
    if rare is not None and rare.any() and (~rare).any():
        out["rare_recall"], out["common_recall"] = float(s["recall"][:, rare].mean()), float(s["recall"][:, ~rare].mean())
    return out


# ---------------------------------------------------------------- the classifier
def fit_classifiers(pools, val, test, n_classes, runs=15, ens=5, max_epochs=10, patience=3, batch=128, lr=1e-4,
                    seed=0, p_syn=0.5, warp=None, device="cuda", log=print, tag="", amp=True, checks_per_epoch=4):
    """Train `runs` independent ensembles of `ens` CNNs side by side and return each run's test predictions.

    pools: one training set, or [real, synthetic]; with two, every training curve is synthetic with
    probability p_syn. Batches are class-balanced inside each pool (the resampling block with gamma = 1),
    one epoch = as many curves as the first pool holds. Each run is checked on the validation stars 4 times
    per epoch and keeps the check with the lowest class-balanced validation loss (early stopping); training
    ends once no run has improved for `patience` epochs. This is astra.classifier.fit_ensembles with two
    changes for the stratified split: the validation loss weights every class equally, and the test
    predictions are returned (for balanced accuracy, bootstrap intervals and paired tests).
    """
    torch.manual_seed(seed)
    M = runs * ens
    model = CNNEnsemble(M, n_classes, pools[0]["x"].shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(enabled=amp_on(amp, device))
    Xtr = torch.cat([stack_input(p["x"], p["ph"]) for p in pools]).to(device)
    ytr = torch.cat([p["y"] for p in pools]).to(device)
    off = len(pools[0]["y"])
    samplers = [[ClassResampler(p["y"].cpu().numpy(), 1.0, seed=seed * 1000 + m + 500_000 * k, n_classes=n_classes)
                 for m in range(M)] for k, p in enumerate(pools)]
    coin = np.random.default_rng(seed + 77)
    Xva, yva = stack_input(val["x"], val["ph"]).to(device), val["y"].cpu().numpy()
    Xte, yte = stack_input(test["x"], test["ph"]).to(device), test["y"].cpu().numpy()
    cnt = np.bincount(yva, minlength=n_classes)
    wva = 1.0 / (np.sum(cnt > 0) * cnt[yva])                           # every class weighs the same, sum = 1
    steps = math.ceil(off / batch / checks_per_epoch)                  # training steps between two checks

    best_loss, since = np.full(runs, np.inf), np.zeros(runs, int)
    best = {"epoch": np.zeros(runs), "val_bal_acc": np.zeros(runs), "val_loss": np.zeros(runs),
            "test_pred": np.zeros((runs, len(yte)), np.int8)}
    curves = {"epoch": [], "train_loss": [], "val_loss": [], "val_bal_acc": []}
    for check in range(1, int(max_epochs * checks_per_epoch) + 1):
        epoch = check / checks_per_epoch
        model.train()
        idx = np.stack([s.sample(steps * batch) for s in samplers[0]])                         # (M, S*B)
        if len(pools) == 2:
            idx_syn = np.stack([s.sample(steps * batch) for s in samplers[1]]) + off
            idx = np.where(coin.random(idx.shape) < p_syn, idx_syn, idx)
        idx = torch.from_numpy(idx).to(device)
        tot = torch.zeros((), device=device)
        for s in range(steps):
            ib = idx[:, s * batch:(s + 1) * batch]
            xb, yb = Xtr[ib], ytr[ib]                                  # (M, B, 2, L), (M, B)
            if warp is not None:
                xb = warp(xb)
            with autocast(device, amp):
                logits = model(xb)
            loss = F.cross_entropy(logits.float().reshape(-1, n_classes), yb.reshape(-1)) * M   # sum over members
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.detach() / M
        tot = tot.item() / steps
        # every run = the average of its members' softmax outputs
        pv = predict_proba(model, Xva, amp=amp).reshape(runs, ens, len(yva), n_classes).mean(1).cpu().numpy()
        vloss = (-np.log(np.take_along_axis(pv, yva[None, :, None], 2)[:, :, 0] + 1e-12) * wva).sum(1)   # (runs,)
        vbal = ((pv.argmax(2) == yva[None]) * wva).sum(1)
        better = vloss < best_loss
        if better.any():                                               # test predictions only when a run improved
            tp = predict_proba(model, Xte, amp=amp).reshape(runs, ens, len(yte), n_classes).mean(1).argmax(2).cpu().numpy()
            best["test_pred"][better] = tp[better]
            best["epoch"][better], best["val_bal_acc"][better], best["val_loss"][better] = epoch, vbal[better], vloss[better]
            best_loss[better] = vloss[better]
        since = np.where(better, 0, since + 1)
        for k, v in zip(curves, (epoch, tot, float(vloss.mean()), float(vbal.mean()))):
            curves[k].append(v)
        if check % checks_per_epoch == 0:
            log(f"  {tag} epoch {epoch:.0f}: train loss {tot:.3f} | val loss {vloss.mean():.3f} "
                f"balanced acc {vbal.mean():.3f}")
        if (since >= patience * checks_per_epoch).all():
            break
    return {"test_pred": best["test_pred"], "val_bal_acc": best["val_bal_acc"].tolist(),
            "val_loss": best["val_loss"].tolist(), "best_epoch": best["epoch"].tolist(), "epochs_run": epoch,
            "curves": curves}


# ---------------------------------------------------------------- fidelity and memorisation
class Fidelity:
    """Feature-space scores of a synthetic set against the real training curves of one split."""

    def __init__(self, data, feat, k=5):
        self.data, self.feat, self.k = data, feat, k
        tr = data["train_idx"]
        f, ytr = extract_features(feat, data["x"][tr], data["ph"][tr]), data["y"][tr]
        self.per = []                                   # per class: real features and their k-NN radii
        for c in range(data["n_classes"]):
            fc = f[ytr == c]
            self.per.append((fc, knn_radii(fc, k)))
        f64 = f.cpu().numpy().astype(np.float64)
        self.real_stats = (f64.mean(0), np.cov(f64, rowvar=False))
        self.prior = data["train_counts"] / data["train_counts"].sum()

    def __call__(self, x_syn, idx):
        d = self.data
        fg, yg = extract_features(self.feat, x_syn, d["ph"][idx]), d["y"][idx]
        out = {m: [] for m in ("precision", "recall", "density", "coverage")}
        for c in range(d["n_classes"]):
            fc = fg[yg == c]
            r = prdc(self.per[c][0], fc, self.k, real_radii=self.per[c][1]) if len(fc) > self.k else {}
            for m in out:
                out[m].append(r.get(m, float("nan")))
        # FID at the natural class mix: the synthetic set is class-balanced, so every curve is weighted by
        # (share of its class in the training set) / (share of its class in this set)
        yg = yg.cpu().numpy()
        w = (self.prior / np.maximum(np.bincount(yg, minlength=d["n_classes"]) / len(yg), 1e-12))[yg]
        w = w / w.sum()
        f64 = fg.cpu().numpy().astype(np.float64)
        mu = (w[:, None] * f64).sum(0)
        cov = (w[:, None] * (f64 - mu)).T @ (f64 - mu)
        try:
            out["fid"] = fid_from_stats(*self.real_stats, mu, cov)
        except Exception:                   # a failed matrix square root: no value, but the job goes on
            out["fid"] = float("nan")
        # the amplitude condition: half the peak-to-peak range of a generated curve should equal it
        half_range = (x_syn.max(1).values - x_syn.min(1).values) / 2
        out["amp_err"] = float((half_range / d["amp"][idx] - 1).abs().mean())
        return out


def own_distance(data, x_syn, idx):
    """RMS difference between a generated curve and the real curve of its condition star, in amplitudes."""
    return ((x_syn - data["x"][idx]) / data["amp"][idx][:, None]).pow(2).mean(1).sqrt()


def memorisation(data, sample, x_syn, idx, n, seed, **opts):
    """Per class: (median own-distance for training conditions / the same for validation conditions,
    share of training-condition curves closer to their source than the 5th percentile of the validation ones)."""
    idx_v = balanced_idx(data, n, seed, side="val")
    d_t, d_v = own_distance(data, x_syn, idx), own_distance(data, sample(idx_v, seed + 1, **opts), idx_v)
    yt, yv = data["y"][idx], data["y"][idx_v]
    ratio, copy = [], []
    for c in range(data["n_classes"]):
        if (yt == c).any() and (yv == c).any():
            a, b = d_t[yt == c], d_v[yv == c]
            ratio.append(float(a.median() / b.median()))
            copy.append(float((a < torch.quantile(b.float(), 0.05)).float().mean()))
        else:
            ratio.append(float("nan")); copy.append(float("nan"))
    return ratio, copy


def fixed_rows(data, side, n, seed=0):
    idx = data[f"{side}_idx_np"]
    pick = np.random.default_rng(seed).choice(len(idx), min(n, len(idx)), replace=False)
    return data[f"{side}_idx"][torch.from_numpy(np.sort(pick)).to(data["device"])]


# ---------------------------------------------------------------- one checkpoint (validation side only)
def evaluate_checkpoint(model, fidelity, data, cfg, k, device):
    t0 = time.time()
    idx = balanced_idx(data, cfg["n_sel"], seed=50_000 + k)
    x = model.sample(idx, seed=60_000 + k)
    res = {"ckpt": k, "nonfinite": model.n_bad, **fidelity(x, idx)}
    res["mem_ratio"], res["copy_share"] = memorisation(data, model.sample, x, idx, cfg["n_mem"], seed=65_000 + k)
    r = fit_classifiers([subset(data, idx, x)], subset(data, data["val_idx"]), subset(data, data["test_idx"]),
                        data["n_classes"], runs=cfg["sel_runs"], ens=cfg["ens"], max_epochs=cfg["sel_max_epochs"],
                        patience=cfg["sel_patience"], seed=k, device=device, log=lambda *a: None,
                        amp=cfg.get("amp", True))
    res["val_bal_acc"], res["val_bal_acc_runs"] = float(np.mean(r["val_bal_acc"])), r["val_bal_acc"]
    # logged for the analysis only: never used to choose a checkpoint
    res["test_bal_acc_info"] = summarise(data["y_np"][data["test_idx_np"]], r["test_pred"], data["n_classes"])["bal_acc"]
    # the model's own loss on training stars and on validation stars (over/under-fitting check)
    res["loss_train"] = model.losses(fixed_rows(data, "train", cfg["n_loss"]), seed=1)
    res["loss_val"] = model.losses(fixed_rows(data, "val", cfg["n_loss"]), seed=1)
    res["eval_s"] = time.time() - t0
    return res


# ---------------------------------------------------------------- the selected checkpoint (test side)
def final_evaluation(model, fidelity, data, cfg, out, deadline, device, log, sweeps=()):
    """Test scores of the selected checkpoint. Every part is saved when it is done, so a session that runs out
    of time continues here in the next one. Returns the dict of parts, or None if it had to stop early."""
    parts = load_json(out / "final_partial.json", {})
    preds = dict(np.load(out / "final_pred.npz")) if (out / "final_pred.npz").exists() else {}
    N, rare = data["n_classes"], data["rare"]
    va, te, real = (subset(data, data[f"{s}_idx"]) for s in ("val", "test", "train"))
    yte = data["y_np"][data["test_idx_np"]]

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    def fit(name, pools, runs, max_epochs, patience, **kw):
        t0 = time.time()
        r = fit_classifiers(pools, va, te, N, runs=runs, ens=cfg["ens"], max_epochs=max_epochs, patience=patience,
                            seed=1, device=device, log=log, tag=name, amp=cfg.get("amp", True), **kw)
        s = summarise(yte, r["test_pred"], N, rare)
        s.update(val_bal_acc=float(np.mean(r["val_bal_acc"])), best_epoch=float(np.mean(r["best_epoch"])),
                 curves=r["curves"], minutes=(time.time() - t0) / 60)
        preds[name] = r["test_pred"]
        log(f"{name}: test balanced accuracy {100 * s['bal_acc']:.2f} +- {100 * s['bal_acc_std']:.2f} | "
            f"macro-F1 {100 * s['macro_f1']:.2f} | val balanced accuracy {100 * s['val_bal_acc']:.2f}")
        return s

    def save():
        save_npz(out / "final_pred.npz", **preds)
        save_json(parts, out / "final_partial.json")

    # one synthetic curve per (balanced) training condition; the same conditions for every family
    idx = balanced_idx(data, cfg.get("n_final") or len(data["train_idx"]), seed=70_000)
    if "samples" not in parts:
        sync(); t0 = time.time()
        x = model.sample(idx, seed=80_000)
        sync(); dt = time.time() - t0
        save_npz(out / "samples.npz", True, idx=idx.cpu().numpy(), x=x.cpu().numpy().astype(np.float16))
        mem, copy = memorisation(data, model.sample, x, idx, cfg["n_mem"], seed=85_000)
        parts["samples"] = {"n": len(idx), "sampling_s_per_10k": dt / len(idx) * 1e4, "nonfinite": model.n_bad,
                            **fidelity(x, idx), "mem_ratio": mem, "copy_share": copy}
        save()
    x = torch.from_numpy(np.load(out / "samples.npz")["x"].astype(np.float32)).to(device)
    syn = subset(data, idx, x)
    fin = dict(runs=cfg["final_runs"], max_epochs=cfg["final_max_epochs"], patience=cfg["final_patience"])
    todo = [("tstr", [syn], {})] + [(f"aug_p{round(100 * p)}", [real, syn], {"p_syn": p}) for p in cfg["p_syn"]]
    for name, pools, kw in todo:
        if name in parts:
            continue
        if deadline.passed(900):
            log("time budget reached during the final evaluation; it continues next session")
            return None
        parts[name] = fit(name, pools, **fin, **kw)
        save()
    # other sampling settings of the same checkpoint (smaller: n_sel curves, 3 runs)
    sel = dict(runs=cfg["sel_runs"], max_epochs=cfg["sel_max_epochs"], patience=cfg["sel_patience"])
    idx_s = balanced_idx(data, cfg["n_sel"], seed=75_000)
    for i, opts in enumerate([{}] + list(sweeps)):
        name = "sweep_" + ("default" if not opts else "_".join(f"{k}{v}" for k, v in opts.items()))
        if name in parts:
            continue
        if deadline.passed(900):
            log("time budget reached during the sampling sweeps; they continue next session")
            return None
        sync(); t0 = time.time()
        xs = model.sample(idx_s, seed=76_000, **opts)
        sync(); dt = time.time() - t0
        parts[name] = {"opts": opts, "sampling_s_per_10k": dt / len(idx_s) * 1e4, "fidelity": fidelity(xs, idx_s),
                       **fit(name, [subset(data, idx_s, xs)], **sel)}
        save()
    return parts


def tstr_only(model, data, cfg, out, deadline, device, log, name):
    """The test score of another checkpoint (the compute-matched budgets): TSTR with the same conditions,
    seeds and classifier runs as the headline. Saved as one more part, so it resumes like the others."""
    parts = load_json(out / "final_partial.json", {})
    if name in parts:
        return parts[name]
    if deadline.passed(900):
        log(f"time budget reached before {name}; it continues next session")
        return None
    preds = dict(np.load(out / "final_pred.npz")) if (out / "final_pred.npz").exists() else {}
    idx = balanced_idx(data, cfg.get("n_final") or len(data["train_idx"]), seed=70_000)
    t0 = time.time()
    x = model.sample(idx, seed=80_000)
    va, te = (subset(data, data[f"{s}_idx"]) for s in ("val", "test"))
    r = fit_classifiers([subset(data, idx, x)], va, te, data["n_classes"], runs=cfg["final_runs"], ens=cfg["ens"],
                        max_epochs=cfg["final_max_epochs"], patience=cfg["final_patience"], seed=1, device=device,
                        log=log, tag=name, amp=cfg.get("amp", True))
    s = summarise(data["y_np"][data["test_idx_np"]], r["test_pred"], data["n_classes"], data["rare"])
    s.update(val_bal_acc=float(np.mean(r["val_bal_acc"])), best_epoch=float(np.mean(r["best_epoch"])),
             minutes=(time.time() - t0) / 60, nonfinite=model.n_bad)
    parts[name], preds[name] = s, r["test_pred"]
    save_npz(out / "final_pred.npz", **preds)
    save_json(parts, out / "final_partial.json")
    log(f"{name}: test balanced accuracy {100 * s['bal_acc']:.2f} +- {100 * s['bal_acc_std']:.2f}")
    return s


# ---------------------------------------------------------------- conditions that no training star has
def novel_conditions(data, idx, seed):
    """For every training-condition row of idx: the phases of ANOTHER training star of the same class and the
    amplitude of a third one; the class is kept. No training star has that combination, so a model that has
    learnt to look up a training curve by its condition cannot copy one; what it generates then shows how much
    of the class it has learnt. Returns a view of the data whose rows 0 .. len(idx)-1 are these conditions (all
    that a family's cond() and to_mag() read), and the two donor rows of every condition."""
    rng = np.random.default_rng(seed)
    rows, y, star, tr = idx.cpu().numpy(), data["y_np"], data["star"], data["train_idx_np"]

    def donors(avoid):
        pick = np.empty_like(rows)
        for c in np.unique(y[rows]):
            k = np.flatnonzero(y[rows] == c)
            pool = tr[y[tr] == c]
            assert len(np.unique(star[pool])) >= 3, f"class {c}: too few training stars for new conditions"
            pick[k] = rng.choice(pool, len(k))
            for _ in range(100):           # another star than every row in `avoid` (redraw the clashes)
                bad = np.any([star[pick[k]] == star[a[k]] for a in avoid], axis=0)
                if not bad.any():
                    break
                pick[k[bad]] = rng.choice(pool, int(bad.sum()))
            assert not np.any([star[pick[k]] == star[a[k]] for a in avoid]), "no different donor star found"
        return pick

    ph_from = donors([rows])
    amp_from = donors([rows, ph_from])
    p, a = (torch.from_numpy(v).to(idx.device) for v in (ph_from, amp_from))
    view = {k: v for k, v in data.items() if k not in ("x", "u")}     # no curves: a family must not read them here
    view.update(ph=data["ph"][p], y=data["y"][idx], onehot=data["onehot"][idx], camp=data["camp"][a],
                amp=data["amp"][a], mag_scale=data["mag_scale"][a])
    return view, ph_from, amp_from


class conditions_of:
    """Inside this block the model generates for the condition rows of `view` instead of its own data."""

    def __init__(self, model, view):
        self.model, self.view = model, view

    def __enter__(self):
        self.old, self.model.data = self.model.data, self.view

    def __exit__(self, *exc):
        self.model.data = self.old


def condition_test(model, data, cfg, out, deadline, device, log, name, novel, ckpt):
    """TSTR of the loaded checkpoint with the sampling sweeps' sizes (n_sel balanced training conditions, sel_runs
    classifier runs, the same seeds): for the training stars' own conditions (at the selected checkpoint this is
    the part sweep_default), or, with novel, for the same rows' new conditions (novel_conditions). No
    memorisation score here: a new condition has no source curve. Saved as one more part, so it resumes."""
    parts = load_json(out / "final_partial.json", {})
    if name in parts:
        return parts[name]
    if deadline.passed(900):
        log(f"time budget reached before {name}; it continues next session")
        return None
    preds = dict(np.load(out / "final_pred.npz")) if (out / "final_pred.npz").exists() else {}
    idx = balanced_idx(data, cfg["n_sel"], seed=75_000)
    t0 = time.time()
    if novel:
        view, _, _ = novel_conditions(data, idx, seed=77_000)
        with conditions_of(model, view):
            x = model.sample(torch.arange(len(idx), device=idx.device), seed=76_000)
        pool = {"x": x, "ph": view["ph"], "y": view["y"]}
    else:
        pool = subset(data, idx, model.sample(idx, seed=76_000))
    va, te = (subset(data, data[f"{s}_idx"]) for s in ("val", "test"))
    r = fit_classifiers([pool], va, te, data["n_classes"], runs=cfg["sel_runs"], ens=cfg["ens"],
                        max_epochs=cfg["sel_max_epochs"], patience=cfg["sel_patience"], seed=1, device=device,
                        log=log, tag=name, amp=cfg.get("amp", True))
    s = summarise(data["y_np"][data["test_idx_np"]], r["test_pred"], data["n_classes"], data["rare"])
    s.update(val_bal_acc=float(np.mean(r["val_bal_acc"])), best_epoch=float(np.mean(r["best_epoch"])),
             minutes=(time.time() - t0) / 60, nonfinite=model.n_bad, ckpt=ckpt, novel=novel)
    parts[name], preds[name] = s, r["test_pred"]
    save_npz(out / "final_pred.npz", **preds)
    save_json(parts, out / "final_partial.json")
    log(f"{name} (checkpoint {ckpt}): test balanced accuracy {100 * s['bal_acc']:.2f} +- {100 * s['bal_acc_std']:.2f}")
    return s


def run_real(data, cfg, out, deadline, device, log):
    """The reference rows on real data only: A = oversampling, B-D = + soft warping (Table 5 of the paper)."""
    out.mkdir(parents=True, exist_ok=True)
    parts = load_json(out / "final_partial.json", {})
    preds = dict(np.load(out / "final_pred.npz")) if (out / "final_pred.npz").exists() else {}
    va, te, real = (subset(data, data[f"{s}_idx"]) for s in ("val", "test", "train"))
    yte = data["y_np"][data["test_idx_np"]]
    for row, mode in REAL_ROWS.items():
        if row in parts:
            continue
        if deadline.passed(900):
            log("time budget reached; the remaining rows run next session")
            return None
        t0 = time.time()
        r = fit_classifiers([real], va, te, data["n_classes"], runs=cfg["final_runs"], ens=cfg["ens"],
                            max_epochs=cfg["final_max_epochs"], patience=cfg["final_patience"], seed=1,
                            warp=make_warp(mode), device=device, log=log, tag=row, amp=cfg.get("amp", True))
        s = summarise(yte, r["test_pred"], data["n_classes"], data["rare"])
        s.update(val_bal_acc=float(np.mean(r["val_bal_acc"])), best_epoch=float(np.mean(r["best_epoch"])),
                 curves=r["curves"], minutes=(time.time() - t0) / 60)
        parts[row], preds[row] = s, r["test_pred"]
        save_npz(out / "final_pred.npz", **preds)
        save_json(parts, out / "final_partial.json")
        log(f"{row}: test balanced accuracy {100 * s['bal_acc']:.2f} +- {100 * s['bal_acc_std']:.2f} | "
            f"macro-F1 {100 * s['macro_f1']:.2f} ({s['minutes']:.1f} min)")
    return parts
