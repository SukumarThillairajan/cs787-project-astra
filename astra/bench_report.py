"""Tables and figures of the ASTRA benchmark, built from the result files the workers write (one per job).

Every benchmark notebook calls this on its own results. If the outputs of the other notebooks are attached
(Add Input), the same calls give the four-way comparison, because all notebooks use the same split, the same
conditions and the same classifier harness (checked through the data hash stored in every result).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILIES = {"real": "real data only", "gan": "GAN", "vae": "VAE", "ddpm": "diffusion (DDPM)", "fm": "flow matching"}
MAIN = ("gan", "vae", "ddpm", "fm")      # every final-stage job of a family is its main (searched) setting


def collect(roots, folder, stage="final"):
    """job -> result of one stage ("final": test scores; "search": validation only), from every <folder>/*.json
    below the given roots. A final result wins over a search result of the same job, then the first root wins."""
    out = {}
    for root in map(Path, roots):
        if not root.exists():
            continue
        for p in sorted(root.glob(f"**/{folder}/*.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            if "job" not in r:
                continue
            r.setdefault("stage", "final" if "parts" in r else "search")
            if r["stage"] != stage or r["job"] in out or (stage == "final" and "parts" not in r):
                continue
            r["_pred"], r["_from"] = p.with_name(p.stem + "_pred.npz"), str(p.parent)
            out[r["job"]] = r
    return out


def rare_mean(r, values):
    """Mean of a per-class list over the rare tier of the result's split (nan if there is none)."""
    v, rare = np.array(values, float), np.array(r["rare"], bool)
    return float(np.nanmean(v[rare])) if rare.any() else float("nan")


COPY_LIMIT = 0.15    # a checkpoint whose rare-tier copy share is above this counts as copying (5% = none)


def budget_hours(key):
    """'30min' -> 0.5 (astra.bench_run.budget_key); results written before 2026-10-22 use hours ('0.5h')."""
    return float(key[:-3]) / 60 if key.endswith("min") else float(key[:-1])


def budget_label(key):
    h = budget_hours(key)
    return f"{60 * h:g} min" if h < 1 else f"{h:g} h"


def bootstrap_bal_acc(y, star, preds, n_classes, n_boot=1000, seed=0):
    """Balanced accuracy (mean over the runs) on n_boot resamples of the test STARS, drawn with replacement;
    all curves of a star stay together. Every entry of preds (runs x n) sees the same resamples, so the
    difference between two entries is a paired comparison."""
    rng = np.random.default_rng(seed)
    stars, inv = np.unique(star, return_inverse=True)
    W = rng.multinomial(len(stars), np.full(len(stars), 1 / len(stars)), size=n_boot).astype(np.float32)[:, inv.reshape(-1)]
    cls = [np.flatnonzero(y == c) for c in range(n_classes)]
    n_true = np.stack([W[:, i].sum(1) for i in cls], 1)                          # (n_boot, N)
    has = n_true > 0
    out = {}
    for name, p in preds.items():
        ok = (np.atleast_2d(p) == y[None]).astype(np.float32)                    # (runs, n)
        rec = np.stack([W[:, i] @ ok[:, i].T for i in cls], 2) / np.maximum(n_true[:, None, :], 1e-9)
        out[name] = ((rec * has[:, None, :]).sum(2) / np.maximum(has.sum(1), 1)[:, None]).mean(1)
    return out


def ci(d):
    return f"{d.mean():+.2f} [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]"


def best_aug(parts):
    """The real + synthetic mix with the best VALIDATION balanced accuracy."""
    augs = {k: v for k, v in parts.items() if k.startswith("aug_p")}
    return max(augs, key=lambda k: augs[k]["val_bal_acc"]) if augs else None


def paired(res, a, b, n_boot=1000):
    """Bootstrap samples (in %) of entries a and b, each given as (job, part name). None if they cannot be paired."""
    (ja, pa), (jb, pb) = a, b
    if ja not in res or jb not in res or res[ja]["data_hash"] != res[jb]["data_hash"]:
        return None
    if not (res[ja]["_pred"].exists() and res[jb]["_pred"].exists()):
        return None
    za, zb = np.load(res[ja]["_pred"]), np.load(res[jb]["_pred"])
    assert np.array_equal(za["y_test"], zb["y_test"]), "the two jobs do not share a test set"
    s = bootstrap_bal_acc(za["y_test"], za["star_test"], {"a": za[pa], "b": zb[pb]}, len(res[ja]["classes"]), n_boot)
    return 100 * s["a"], 100 * s["b"]


def job_table(res, n_boot=1000):
    """One row per generative job: scores of the selected checkpoint and its difference to real data only."""
    rows = []
    for job, r in sorted(res.items()):
        if r["family"] == "real":
            continue
        P, S = r["parts"], r["parts"]["samples"]
        aug, base = best_aug(P), f"real_s{r['seed']}"
        row = {"job": job, "ckpt": r["selected_ckpt"], "selected at (h)": round(r.get("selected_solo_h", np.nan), 2),
               "trained (h)": round(r.get("solo_h", np.nan), 2), "stopped by": r.get("stop", ""),
               "TSTR bal-acc": f"{100 * P['tstr']['bal_acc']:.2f} ± {100 * P['tstr']['bal_acc_std']:.2f}",
               "copies, rare tier %": round(100 * rare_mean(r, S.get("copy_share", [np.nan] * len(r["rare"]))), 1),
               "mem ratio, rare tier": round(rare_mean(r, S["mem_ratio"]), 2),
               "TSTR macro-F1": round(100 * P["tstr"]["macro_f1"], 2)}
        if aug:
            row.update({"real+syn bal-acc": f"{100 * P[aug]['bal_acc']:.2f} ± {100 * P[aug]['bal_acc_std']:.2f}",
                        "p_syn": int(aug[5:]) / 100, "real+syn rare recall": round(100 * P[aug].get("rare_recall", np.nan), 2)})
        if base in res:
            row["real only (A)"] = round(100 * res[base]["parts"]["A_oversampling"]["bal_acc"], 2)
            for part, col in (("tstr", "TSTR - A [95% CI]"), (aug, "real+syn - A [95% CI]")):
                s = paired(res, (job, part), (base, "A_oversampling"), n_boot) if part else None
                row[col] = ci(s[0] - s[1]) if s else ""
        row.update({"FID": round(S["fid"], 2), "density": round(float(np.nanmean(S["density"])), 3),
                    "recall": round(float(np.nanmean(S["recall"])), 3), "coverage": round(float(np.nanmean(S["coverage"])), 3),
                    "mem ratio (all)": round(float(np.nanmean(S["mem_ratio"])), 2), "amp err": round(S["amp_err"], 3),
                    "s per 10k curves": round(S["sampling_s_per_10k"], 2)})
        rows.append(row)
    return pd.DataFrame(rows)


def real_table(res):
    """The reference rows on real data (A = oversampling, B-D = soft warping), one line per split."""
    rows = []
    for job, r in sorted(res.items()):
        if r["family"] != "real":
            continue
        for name, p in r["parts"].items():
            rows.append({"job": job, "row": name, "bal-acc": f"{100 * p['bal_acc']:.2f} ± {100 * p['bal_acc_std']:.2f}",
                         "macro-F1": round(100 * p["macro_f1"], 2), "accuracy": round(100 * p["acc"], 2),
                         "rare recall": round(100 * p.get("rare_recall", np.nan), 2),
                         "common recall": round(100 * p.get("common_recall", np.nan), 2),
                         "epochs (best)": round(p["best_epoch"], 1), "from": Path(r["_from"]).parent.name})
    return pd.DataFrame(rows)


def family_table(res):
    """Mean ± standard deviation over the splits (seeds) of the main setting of every family present."""
    def agg(vals):
        v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
        return f"{v.mean():.2f} ± {v.std():.2f}" if len(v) else ""

    rows = []
    real = [r for r in res.values() if r["family"] == "real"]
    for name in ("A_oversampling", "B_soft_time_warping", "C_soft_mag_warping", "D_soft_mixed_warping"):
        got = [r["parts"][name] for r in real if name in r["parts"]]
        if got:
            rows.append({"method": f"real only, {name[2:].replace('_', ' ')}", "splits": len(got),
                         "bal-acc": agg(100 * p["bal_acc"] for p in got), "macro-F1": agg(100 * p["macro_f1"] for p in got),
                         "rare recall": agg(100 * p.get("rare_recall", np.nan) for p in got),
                         "common recall": agg(100 * p.get("common_recall", np.nan) for p in got)})
    for fam in MAIN:
        rs = [r for _, r in main_jobs(res, fam)]
        got = [r["parts"] for r in rs]
        if not got:
            continue
        fidelity = {"copies, rare tier %": agg(100 * rare_mean(r, r["parts"]["samples"].get("copy_share", [np.nan] * len(r["rare"])))
                                               for r in rs),
                    "mem ratio, rare tier": agg(rare_mean(r, r["parts"]["samples"]["mem_ratio"]) for r in rs),
                    "FID": agg(P["samples"]["fid"] for P in got),
                    "recall (diversity)": agg(float(np.nanmean(P["samples"]["recall"])) for P in got),
                    "density (fidelity)": agg(float(np.nanmean(P["samples"]["density"])) for P in got),
                    "trained (h, one T4)": agg(r.get("solo_h", np.nan) for r in rs)}
        for label, pick, extra in ((f"{FAMILIES[fam]}, synthetic only (TSTR)", lambda P: P["tstr"], fidelity),
                                   (f"{FAMILIES[fam]}, real + synthetic", lambda P: P[best_aug(P)] if best_aug(P) else None, {})):
            ps = [pick(P) for P in got if pick(P)]
            if ps:
                rows.append({"method": label, "splits": len(ps), "bal-acc": agg(100 * p["bal_acc"] for p in ps),
                             "macro-F1": agg(100 * p["macro_f1"] for p in ps),
                             "rare recall": agg(100 * p.get("rare_recall", np.nan) for p in ps),
                             "common recall": agg(100 * p.get("common_recall", np.nan) for p in ps), **extra})
    return pd.DataFrame(rows).fillna("")


def class_table(res):
    """Per-class test recall (%), mean over the splits; classes from most to least common."""
    cols = {}
    real = [r["parts"]["A_oversampling"]["recall"] for r in res.values()
            if r["family"] == "real" and "A_oversampling" in r["parts"]]
    if real:
        cols["real only (A)"] = 100 * np.mean(real, 0)
    for fam in MAIN:
        got = [r["parts"] for _, r in main_jobs(res, fam)]
        if got:
            cols[f"{FAMILIES[fam]} TSTR"] = 100 * np.mean([P["tstr"]["recall"] for P in got], 0)
            if best_aug(got[0]):
                cols[f"{FAMILIES[fam]} real+syn"] = 100 * np.mean([P[best_aug(P)]["recall"] for P in got], 0)
    if not cols:
        return pd.DataFrame()
    r0 = next(iter(res.values()))
    t = pd.DataFrame(cols, index=r0["classes"]).round(1)
    t.insert(0, "train curves", r0["train_counts"])
    return t


def family_pairs(res, n_boot=1000):
    """Paired comparison of every two families on the splits both have (main settings only)."""
    rows = []
    fams = [f for f in MAIN if main_jobs(res, f)]
    for i, fa in enumerate(fams):
        for fb in fams[i + 1:]:
            ja_by, jb_by = ({s: r["job"] for s, r in main_jobs(res, f)} for f in (fa, fb))
            for seed in sorted(set(ja_by) & set(jb_by)):
                ja, jb = ja_by[seed], jb_by[seed]
                row = {"A": FAMILIES[fa], "B": FAMILIES[fb], "split": seed}
                for label, pa, pb in (("TSTR: A - B [95% CI]", "tstr", "tstr"),
                                      ("real+syn: A - B [95% CI]", best_aug(res[ja]["parts"]), best_aug(res[jb]["parts"]))):
                    s = paired(res, (ja, pa), (jb, pb), n_boot) if pa and pb else None
                    row[label] = ci(s[0] - s[1]) if s else "(different data or no predictions)"
                rows.append(row)
    return pd.DataFrame(rows)


def novel_table(res):
    """Does a model that copies still generate useful curves for conditions it has not seen? Per final job: test
    TSTR (n_sel curves, 3 classifier runs) at the selected and at the last checkpoint, for the training stars'
    own conditions and for new ones (phases of one star, amplitude of another, same class)."""
    rows = []
    for job, r in sorted(res.items()):
        P = r.get("parts", {})
        if "novel_selected" not in P:
            continue
        cells = {"selected": ("sweep_default", "novel_selected"), "last": ("default_last", "novel_last")}
        for which, (own, new) in cells.items():
            if own not in P and new not in P:
                continue
            k = r.get(f"{which}_ckpt")
            sel = next((a for a in r.get("metrics", []) if a["ckpt"] == k), {})
            row = {"job": job, "checkpoint": f"{which} ({k}, {60 * r.get(f'{which}_solo_h', np.nan):.0f} min)",
                   "copies, rare tier % (validation)": round(100 * rare_mean(r, sel.get("copy_share", [np.nan] * len(r["rare"]))), 1)}
            for label, name in (("own conditions", own), ("new conditions", new)):
                p = P.get(name)
                row[f"{label}: test bal-acc"] = round(100 * p["bal_acc"], 2) if p else np.nan
                row[f"{label}: rare recall"] = round(100 * p.get("rare_recall", np.nan), 2) if p else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def sweep_table(res):
    rows = []
    for job, r in sorted(res.items()):
        for name, p in r["parts"].items():
            if name.startswith("sweep_"):
                rows.append({"job": job, "sampling": "default" if not p["opts"] else ", ".join(f"{k} = {v}" for k, v in p["opts"].items()),
                             "val bal-acc": round(100 * p["val_bal_acc"], 2), "test bal-acc": round(100 * p["bal_acc"], 2),
                             "FID": round(p["fidelity"]["fid"], 2),
                             "density": round(float(np.nanmean(p["fidelity"]["density"])), 3),
                             "recall": round(float(np.nanmean(p["fidelity"]["recall"])), 3),
                             "s per 10k curves": round(p["sampling_s_per_10k"], 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def plot_training(res, path=None):
    """Per checkpoint: validation TSTR balanced accuracy, FID, recall and density, memorisation ratio, and the
    model's own loss on training and validation stars (over/under-fitting)."""
    jobs = [(j, r) for j, r in sorted(res.items()) if r.get("metrics")]
    if not jobs:
        return
    fig, ax = plt.subplots(1, 5, figsize=(24, 3.8))
    for j, r in jobs:
        m = r["metrics"]
        x = [a.get("solo_h", a["ckpt"]) for a in m]
        ax[0].plot(x, [100 * a["val_bal_acc"] for a in m], "o-", ms=3, label=j)
        sel = r.get("selected_ckpt") or r.get("best_ckpt")
        for a in m:
            if a["ckpt"] == sel:
                ax[0].plot([a.get("solo_h", a["ckpt"])], [100 * a["val_bal_acc"]], "r*", ms=12)
        ax[1].plot(x, [a["fid"] for a in m], "o-", ms=3, label=j)
        line, = ax[2].plot(x, [np.nanmean(a["recall"]) for a in m], "o-", ms=3, label=f"{j} recall")
        ax[2].plot(x, [np.nanmean(a["density"]) for a in m], "s--", ms=3, c=line.get_color(), label=f"{j} density")
        line, = ax[3].plot(x, [100 * rare_mean(r, a.get("copy_share", [np.nan] * len(r["rare"]))) for a in m], "o-", ms=3,
                           label=f"{j} copies (rare tier)")
        line, = ax[4].plot(x, [a["loss_train"] for a in m], "o-", ms=3, label=f"{j} train")
        ax[4].plot(x, [a["loss_val"] for a in m], "s--", ms=3, c=line.get_color(), label=f"{j} val")
    ax[3].axhline(5, c="k", lw=0.6)
    for a, t in zip(ax, ("validation TSTR balanced accuracy % (star = selected)", "FID (lower = closer to real)",
                         "macro recall (diversity) and density (fidelity)",
                         "copies of the source curve, rare tier % (5 = none)",
                         "the model's own loss: training vs validation stars")):
        a.set_title(t, fontsize=9); a.set_xlabel("hours of training on one T4"); a.set_xscale("log")
        a.legend(fontsize=6)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    plt.show()


def plot_samples(run_dir, Z, title, path=None, per_class=2):
    """Real curves and generated curves with the same condition (phases, amplitude, class), like Fig. 9."""
    s = Path(run_dir) / "samples.npz"
    if not s.exists():
        return
    S = np.load(s)
    classes = [str(c) for c in Z["classes"]]
    fig, ax = plt.subplots(per_class, len(classes), figsize=(3.4 * len(classes), 2.4 * per_class), sharex=True, squeeze=False)
    for c, name in enumerate(classes):
        for r, i in enumerate(np.flatnonzero(Z["y"][S["idx"]] == c)[:per_class]):
            k = S["idx"][i]
            ax[r, c].plot(Z["phase"][k], Z["mag"][k], ".", ms=3, label="real")
            ax[r, c].plot(Z["phase"][k], S["x"][i].astype(float), ".", ms=3, label="generated")
            ax[r, c].invert_yaxis(); ax[r, c].set_title(name, fontsize=9)
    ax[0, 0].legend(fontsize=7); fig.suptitle(title); fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    plt.show()


def plot_classifiers(res, path=None):
    """Training and validation loss of the final classifiers (over/under-fitting of the harness itself)."""
    panels = [(f"{j}: {name}", p["curves"]) for j, r in sorted(res.items()) for name, p in r["parts"].items()
              if name in ("A_oversampling", "tstr") or name == best_aug(r["parts"])]
    if not panels:
        return
    fig, ax = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3), squeeze=False)
    for a, (title, c) in zip(ax[0], panels):
        a.plot(c["epoch"], c["train_loss"], label="train"); a.plot(c["epoch"], c["val_loss"], label="val (class-balanced)")
        a.set_title(title, fontsize=8); a.set_xlabel("epoch"); a.legend(fontsize=6)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=100)
    plt.show()


def plot_sweeps(res, path=None):
    """Quality against sampling cost for the other sampling settings of the selected checkpoints."""
    t = sweep_table(res)
    if not len(t):
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for job, g in t.groupby("job"):
        ax.scatter(g["s per 10k curves"], g["test bal-acc"], label=job, s=25)
        for _, row in g.iterrows():
            ax.annotate(row["sampling"], (row["s per 10k curves"], row["test bal-acc"]), fontsize=6, xytext=(3, 3),
                        textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("seconds to generate 10,000 curves (log scale)")
    ax.set_ylabel("TSTR test balanced accuracy % (3 runs)"); ax.legend(fontsize=7)
    ax.set_title("sampling settings of the selected checkpoint", fontsize=9); fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    plt.show()


# ---------------------------------------------------------------- figures and scores for the four-way comparison
def main_jobs(res, fam):
    """The finished final-stage jobs of a family (its searched setting), one per split: [(seed, result), ...]."""
    return sorted(((r["seed"], r) for r in res.values() if r["family"] == fam and r.get("stage", "final") == "final"),
                  key=lambda t: t[0])


# ---------------------------------------------------------------- the search (validation only) and the budgets
def search_table(res):
    """One row per searched configuration (split 0, validation stars only), best first within a family.
    The rule for the winner: the highest mean of its three best checkpoints' validation scores (less noisy than
    the single best), read against the copy share: "top-3, no copies" counts only checkpoints whose rare-tier
    copy share is at most COPY_LIMIT (n = how many; group decision, 2026-10-22, after the GAN search, where copying
    checkpoints made up the top-3 of the rule's winner); within about 1 point the cheaper run wins."""
    rows = []
    for job, r in res.items():
        m = r.get("metrics") or []
        b = next((a for a in m if a["ckpt"] == r.get("best_ckpt")), None)
        if b is None:
            continue
        top3 = sorted((a["val_bal_acc"] for a in m), reverse=True)[:3]
        clean = sorted((a["val_bal_acc"] for a in m if rare_mean(r, a["copy_share"]) <= COPY_LIMIT), reverse=True)[:3]
        rows.append({"family": r["family"], "config": r.get("tag") or job,
                     "top-3 mean (rule)": round(100 * float(np.mean(top3)), 2),
                     "top-3, no copies (n)": f"{100 * np.mean(clean):.2f} ({len(clean)})" if clean else "-",
                     "best val bal-acc":
                     round(100 * b["val_bal_acc"], 2), "± (3 runs)": round(100 * float(np.std(b["val_bal_acc_runs"])), 2),
                     "at (h, one T4)": round(b.get("solo_h", np.nan), 2), "trained (h)": round(r.get("solo_h", np.nan), 2),
                     "stopped by": r.get("stop", ""), "copies, rare tier %": round(100 * rare_mean(r, b["copy_share"]), 1),
                     "mem ratio, rare tier": round(rare_mean(r, b["mem_ratio"]), 2), "FID": round(b["fid"], 2),
                     "recall": round(float(np.nanmean(b["recall"])), 3), "checkpoints": len(m)})
    t = pd.DataFrame(rows)
    return t.sort_values(["family", "top-3 mean (rule)"], ascending=[True, False]) if len(t) else t


def plot_search(res, path=None):
    """Validation TSTR balanced accuracy and rare-tier copies of every searched configuration against hours."""
    fams = sorted({r["family"] for r in res.values()})
    if not fams:
        return
    fig, ax = plt.subplots(2, len(fams), figsize=(6 * len(fams), 7), squeeze=False)
    for k, fam in enumerate(fams):
        for job, r in sorted(res.items()):
            if r["family"] != fam or not r.get("metrics"):
                continue
            m = r["metrics"]
            x = [a["solo_h"] for a in m]
            ax[0, k].plot(x, [100 * a["val_bal_acc"] for a in m], "o-", ms=3, label=r.get("tag") or job)
            ax[1, k].plot(x, [100 * rare_mean(r, a["copy_share"]) for a in m], "o-", ms=3, label=r.get("tag") or job)
        ax[0, k].set_title(f"{FAMILIES[fam]}: validation TSTR balanced accuracy %", fontsize=9)
        ax[1, k].set_title(f"{FAMILIES[fam]}: copies of the source curve, rare tier % (5 = none)", fontsize=9)
        ax[1, k].axhline(5, c="k", lw=0.6)
        for a in ax[:, k]:
            a.set_xscale("log"); a.set_xlabel("hours of training on one T4"); a.legend(fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    plt.show()


def budget_table(res):
    """The compute-matched view: per family and budget, the test balanced accuracy (TSTR) of the best checkpoint
    reached within that much training on one T4; mean ± std over the splits. 'full' = no limit."""
    rows = []
    for fam in MAIN:
        jobs = main_jobs(res, fam)
        if not jobs:
            continue
        keys = sorted({k for _, r in jobs for k in r.get("budgets", {})}, key=budget_hours)
        for key in keys + ["full"]:
            vals, at, copies = [], [], []
            for _, r in jobs:
                if key == "full":         # the selected checkpoint; copies measured as for the budgets (validation side)
                    sel = next((a for a in r.get("metrics", []) if a["ckpt"] == r.get("selected_ckpt")), {})
                    vals.append(100 * r["parts"]["tstr"]["bal_acc"]); at.append(r.get("selected_solo_h", np.nan))
                    copies.append(100 * rare_mean(r, sel.get("copy_share", [np.nan] * len(r["rare"]))))
                    continue
                b = r.get("budgets", {}).get(key)
                if b and b["test"] in r["parts"]:
                    vals.append(100 * r["parts"][b["test"]]["bal_acc"]); at.append(b["solo_h"])
                    copies.append(100 * rare_mean(r, b["copy_share"]))
            if vals:
                rows.append({"family": FAMILIES[fam], "budget (one T4)": key if key == "full" else budget_label(key),
                             "splits": len(vals),
                             "test bal-acc (TSTR)": f"{np.mean(vals):.2f} ± {np.std(vals):.2f}",
                             "checkpoint at (h)": round(float(np.mean(at)), 2),
                             "copies, rare tier % (validation)": round(float(np.nanmean(copies)), 1)})
    return pd.DataFrame(rows)


def plot_budget(res, path=None):
    """Test balanced accuracy (TSTR) of each family if training had stopped after each budget (5 min ... 10 h),
    and at its own best point; the dashed line is real data only (row A)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    drawn = False
    for i, fam in enumerate(MAIN):
        jobs = main_jobs(res, fam)
        if not jobs:
            continue
        pts = {}
        for _, r in jobs:
            for key, b in r.get("budgets", {}).items():
                if b["test"] in r["parts"]:
                    pts.setdefault(budget_hours(key), []).append(100 * r["parts"][b["test"]]["bal_acc"])
        if not pts:
            continue
        xs = sorted(pts)
        ax.errorbar(xs, [np.mean(pts[x]) for x in xs], [np.std(pts[x]) for x in xs], fmt="o-", c=f"C{i}",
                    capsize=3, label=FAMILIES[fam])
        full = [(r.get("selected_solo_h", np.nan), 100 * r["parts"]["tstr"]["bal_acc"]) for _, r in jobs]
        ax.plot([f[0] for f in full], [f[1] for f in full], "*", c=f"C{i}", ms=11)
        drawn = True
    real = [100 * r["parts"]["A_oversampling"]["bal_acc"] for r in res.values()
            if r["family"] == "real" and "A_oversampling" in r["parts"]]
    if real:
        ax.axhline(np.mean(real), c="k", ls="--", lw=1, label="real data only (A)")
    if not drawn:
        plt.close(fig)
        return
    ax.set_xscale("log"); ax.set_xlabel("training budget, hours on one T4"); ax.set_ylabel("test balanced accuracy %")
    ax.set_title("if every model had stopped after this much training (stars = own best point)", fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    plt.show()


def samples_of(r):
    """The synthetic curves of a job's selected checkpoint (saved next to its result file), or None."""
    p = r["_pred"].with_name(f"{r['job']}_samples.npz")
    return np.load(p) if p.exists() else None


def plot_headline(res, path=None):
    """Test balanced accuracy of every method: one point per split, a bar for the mean. The dashed line is
    real data only (row A); a method has to be above it to count as useful augmentation."""
    entries = []
    real = sorted((r["seed"], 100 * r["parts"]["A_oversampling"]["bal_acc"]) for r in res.values()
                  if r["family"] == "real" and "A_oversampling" in r["parts"])
    if real:
        entries.append(("real only (A)", [v for _, v in real], "0.5"))
    for i, fam in enumerate(f for f in MAIN if main_jobs(res, f)):
        jobs = main_jobs(res, fam)
        entries.append((f"{FAMILIES[fam]}\nsynthetic only", [100 * r["parts"]["tstr"]["bal_acc"] for _, r in jobs], f"C{i}"))
        aug = [100 * r["parts"][best_aug(r["parts"])]["bal_acc"] for _, r in jobs if best_aug(r["parts"])]
        if aug:
            entries.append((f"{FAMILIES[fam]}\nreal + synthetic", aug, f"C{i}"))
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(1.5 * len(entries) + 2, 4.2))
    for k, (label, vals, color) in enumerate(entries):
        ax.bar(k, np.mean(vals), color=color, alpha=0.35 if "synthetic only" in label else 0.8, width=0.7)
        ax.plot([k] * len(vals), vals, "k.", ms=7)
        ax.text(k, max(vals) + 0.4, f"{np.mean(vals):.1f}", ha="center", fontsize=8)
    if real:
        ax.axhline(np.mean([v for _, v in real]), color="k", ls="--", lw=1)
    ax.set_xticks(range(len(entries))); ax.set_xticklabels([e[0] for e in entries], fontsize=8)
    lo = min(min(e[1]) for e in entries)
    ax.set_ylim(max(0, lo - 5), None); ax.set_ylabel("test balanced accuracy %")
    ax.set_title("balanced accuracy per method (points = splits, bar = mean)", fontsize=9); fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    plt.show()


def plot_class_bars(res, path=None):
    """Per-class test recall of real data only against real + synthetic of every family (mean over splits)."""
    t = class_table(res)
    cols = [c for c in t.columns if c == "real only (A)" or c.endswith("real+syn")] if len(t) else []
    if len(cols) < 2:
        return
    rare = np.array(next(iter(res.values()))["rare"])
    fig, ax = plt.subplots(figsize=(1.6 * len(t) + 3, 4))
    w = 0.8 / len(cols)
    for k, c in enumerate(cols):
        ax.bar(np.arange(len(t)) + (k - (len(cols) - 1) / 2) * w, t[c], w, label=c, color="0.5" if k == 0 else f"C{k - 1}")
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels([f"{n}\n({c:,} train){' *' if r else ''}" for n, c, r in zip(t.index, t["train curves"], rare)], fontsize=8)
    ax.set_ylabel("test recall %"); ax.legend(fontsize=7, ncol=2)
    ax.set_title("recall per class, classes from most to least common (* = rare tier)", fontsize=9); fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    plt.show()


def plot_family_samples(res, Z, path=None, seed=0):
    """One real curve per class and, next to it, the curve every family generated for the SAME condition
    (same star's phases, amplitude and class). Possible because all notebooks use the same conditions."""
    S = {fam: samples_of(r) for fam in MAIN for s, r in main_jobs(res, fam) if s == seed}
    S = {fam: s for fam, s in S.items() if s is not None}
    if not S:
        return
    idx = next(iter(S.values()))["idx"]
    S = {fam: s for fam, s in S.items() if np.array_equal(s["idx"], idx)}        # same conditions only
    classes = [str(c) for c in Z["classes"]]
    fig, ax = plt.subplots(len(classes), 1 + len(S), figsize=(2.9 * (1 + len(S)), 1.9 * len(classes)), sharex=True, squeeze=False)
    for c, name in enumerate(classes):
        rows = np.flatnonzero(Z["y"][idx] == c)
        if not len(rows):
            continue
        i, k = rows[0], idx[rows[0]]
        curves = [("real", Z["mag"][k], "0.2")] + [(FAMILIES[f], s["x"][i].astype(float), f"C{n}") for n, (f, s) in enumerate(S.items())]
        lim = 1.6 * float(np.abs(Z["mag"][k]).max()) + 1e-3
        for a, (label, y, color) in zip(ax[c], curves):
            a.plot(Z["phase"][k], y, ".", ms=3, color=color)
            a.set_ylim(lim, -lim)
            if c == 0:
                a.set_title(label, fontsize=9)
        ax[c, 0].set_ylabel(name, fontsize=8)
    fig.supxlabel("phase"); fig.suptitle(f"split {seed}: a real curve and the generated curve for its condition, per family", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    plt.show()


def fidelity_one_space(res, npz_path, work_dir, device="cpu", log=print):
    """FID, density, recall, coverage and precision of every job's saved samples, all measured with ONE feature
    CNN per split (the one in work_dir, trained there if missing). Each notebook trains its own feature CNN,
    so the fidelity columns of different notebooks are only roughly comparable; these are comparable exactly."""
    import torch
    from . import bench_data, bench_run
    from .bench_eval import Fidelity
    rows = []
    for seed in sorted({r["seed"] for r in res.values() if r["family"] != "real"}):
        data, fid = bench_data.load(npz_path, seed, device), None
        for job, r in sorted(res.items()):
            S = samples_of(r) if r["family"] != "real" and r["seed"] == seed else None
            if S is None or r["data_hash"] != data["hash"]:
                continue
            if fid is None:
                fid = Fidelity(data, bench_run.feature_cnn(Path(work_dir), data, device, log=log))
            f = fid(torch.from_numpy(S["x"].astype(np.float32)).to(device), torch.from_numpy(S["idx"]).to(device))
            rows.append({"job": job, "FID": round(f["fid"], 2), **{m: round(float(np.nanmean(f[m])), 3)
                         for m in ("density", "recall", "coverage", "precision")}, "feature CNN": data["feat_hash"]})
    return pd.DataFrame(rows)
