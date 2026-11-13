"""Figures of the final report that are drawn from result tables (no training, CPU only, about 10 s).

Usage (project root):
  python build/final_figs.py report_final/figures
Writes:
  fig_imbalance.png  class counts after pre-processing, ours vs the GAN paper, Catalina and ZTF (log scale)
  fig_per_class.png  test recall per class, classifier trained on synthetic curves only (TSTR), 5-split means
  fig_forest.png     paired differences on the same split with 95% bootstrap intervals (TSTR)
  fig_pelsvae.png    PELS-VAE replication: validation loss, latent codes vs the prior, latent regression
Inputs: Outputs/Benchmark/comparison_catalina/{jobs,pairs,per_class_recall}.csv and
Outputs/VAE Replication/results-astra-02/{runs,tables}/.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "Outputs" / "Benchmark" / "comparison_catalina"
VAE = ROOT / "Outputs" / "VAE Replication" / "results-astra-02"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.2,
                     "ytick.labelsize": 7.2, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.03})
COL = {"real": "#7f7f7f", "GAN": "#1f77b4", "VAE": "#ff7f0e", "DDPM": "#2ca02c", "FM": "#d62728"}
NAME = {"real": "real only (A)", "GAN": "GAN", "VAE": "VAE", "DDPM": "DDPM", "FM": "flow matching"}
INK, RARE_BG = "#22303f", "#fbeee6"
CLASSES = ["EBSD/D", "RRL", "EBC", "LPV", "DSCT", "CEP"]


def ci(s):
    """'+6.14 [+3.65, +8.82]' -> (6.14, 3.65, 8.82)"""
    v = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", s)]
    return v[0], v[1], v[2]


def fig_imbalance(out):
    # ours: notes/baselines.md (class-range filter k = 2) and the rebuilt ZTF set; paper: its Table 2
    cat = dict(classes=CLASSES, paper=[28980, 7533, 4500, 483, 241, 182], ours=[30090, 7841, 4564, 495, 246, 181],
               train=[21104, 5491, 3194, 347, 172, 127])
    ztf = dict(classes=["EB", "RRL", "LPV", "DSCT", "CEP"], paper=[31477, 18729, 5245, 507, 471],
               ours=[31183, 18729, 5245, 507, 471])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw=dict(width_ratios=[6, 5]))
    for ax, d, title in ((axes[0], cat, "Catalina (benchmark and replication)"),
                         (axes[1], ztf, "ZTF (replication only)")):
        n = len(d["classes"])
        x = np.arange(n)
        rare = np.array(d["ours"]) < 0.05 * sum(d["ours"])
        for i in range(n):
            if rare[i]:
                ax.axvspan(i - 0.5, i + 0.5, color=RARE_BG, lw=0, zorder=0)
        w = 0.27 if "train" in d else 0.38
        offs = (-w, 0, w) if "train" in d else (-w / 2, w / 2)
        ax.bar(x + offs[0], d["paper"], w, color="#c9d3de", edgecolor="#8796a8", lw=0.5, label="paper")
        ax.bar(x + offs[1], d["ours"], w, color=INK, label="ours, after pre-processing")
        for i, v in enumerate(d["ours"]):
            ax.text(x[i] + offs[1], v * 1.15, f"{v:,}", ha="center", va="bottom", fontsize=6.0, color=INK,
                    rotation=90 if v > 9999 else 0)
        if "train" in d:
            ax.bar(x + offs[2], d["train"], w, color="#5b8fc7", label="ours, training stars of split 0")
        ax.set_yscale("log")
        ax.set_ylim(80, 3e5)
        ax.set_xticks(x, d["classes"])
        ax.set_title(title, loc="left")
        ratio = max(d["ours"]) / min(d["ours"])
        ax.text(0.99, 0.97, f"largest / smallest = {ratio:.0f} : 1", transform=ax.transAxes, ha="right",
                va="top", fontsize=6.8, color=INK)
    axes[0].set_ylabel("light curves (log scale)")
    h, lab = axes[0].get_legend_handles_labels()
    h.append(Patch(color=RARE_BG)); lab.append("rare tier (< 5% of the curves)")
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.5)
    fig.savefig(out / "fig_imbalance.png")
    plt.close(fig)


def fig_per_class(out):
    t = pd.read_csv(BENCH / "per_class_recall.csv", index_col=0)
    cols = {"real": "real only (A)", "GAN": "GAN TSTR", "VAE": "VAE TSTR", "DDPM": "diffusion (DDPM) TSTR",
            "FM": "flow matching TSTR"}
    fig, ax = plt.subplots(figsize=(7.2, 2.45))
    x = np.arange(len(CLASSES))
    w = 0.16
    for i in range(3, 6):
        ax.axvspan(i - 0.5, i + 0.5, color=RARE_BG, lw=0, zorder=0)
    for k, (fam, c) in enumerate(cols.items()):
        v = t.loc[CLASSES, c].values.astype(float)
        ax.bar(x + (k - 2) * w, v, w, color=COL[fam], label=NAME[fam], zorder=2)
        if fam in ("real", "FM"):
            for i in (3, 4, 5):
                ax.text(x[i] + (k - 2) * w, v[i] + 1.2, f"{v[i]:.0f}", ha="center", va="bottom", fontsize=5.8,
                        color=COL[fam])
    tr = t.loc[CLASSES, "train curves"].values
    ax.set_xticks(x, [f"{c}\n{int(n):,} train" for c, n in zip(CLASSES, tr)])
    ax.set_ylabel("test recall %")
    ax.set_ylim(0, 104)
    ax.text(4.0, 101, "rare tier", ha="center", va="bottom", fontsize=7, color="#a0522d")
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.04), frameon=False)
    ax.grid(axis="y", color="#e6e9ee", lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(out / "fig_per_class.png")
    plt.close(fig)


def fig_forest(out):
    jobs = pd.read_csv(BENCH / "jobs.csv")
    pairs = pd.read_csv(BENCH / "pairs.csv")
    fam_of = {"gan": "GAN", "vae": "VAE", "ddpm": "DDPM", "fm": "FM"}
    long = {"GAN": "GAN", "VAE": "VAE", "diffusion (DDPM)": "DDPM", "flow matching": "FM"}
    rows = []                                    # (label, colour family, [(d, lo, hi) per split])
    for fam in ("FM", "DDPM", "GAN", "VAE"):
        sub = jobs[jobs.job.str.split("_").str[0].map(fam_of) == fam].sort_values("job")
        rows.append((f"{NAME[fam]} $-$ real only", fam, [ci(s) for s in sub["TSTR - A [95% CI]"]]))
    for a, b in (("FM", "GAN"), ("DDPM", "GAN"), ("FM", "DDPM"), ("FM", "VAE"), ("DDPM", "VAE"), ("GAN", "VAE")):
        vals = []
        for s in range(5):
            r = pairs[(pairs.A.map(long) == a) & (pairs.B.map(long) == b) & (pairs.split == s)]
            sign = 1
            if r.empty:
                r = pairs[(pairs.A.map(long) == b) & (pairs.B.map(long) == a) & (pairs.split == s)]
                sign = -1
            d, lo, hi = ci(r.iloc[0]["TSTR: A - B [95% CI]"])
            vals.append((sign * d, *sorted((sign * lo, sign * hi))))
        rows.append((f"{NAME[a]} $-$ {NAME[b]}", a, vals))
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    yy = []
    y = 0.0
    for k, (lab, fam, vals) in enumerate(rows):
        if k == 4:
            y -= 0.6                                       # gap between the two groups
        yy.append(y)
        for s, (d, lo, hi) in enumerate(vals):
            yo = y + (s - 2) * 0.13
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [yo, yo], color=COL[fam], lw=1.0, alpha=0.9 if sig else 0.45)
            ax.plot(d, yo, "o", ms=3.2, color=COL[fam], mfc=COL[fam] if sig else "white", mew=0.9)
        n_sig = sum(lo > 0 or hi < 0 for _, lo, hi in vals)
        n_pos = sum(lo > 0 for _, lo, hi in vals)
        n_neg = sum(hi < 0 for _, lo, hi in vals)
        txt = f"{n_pos}/5 above 0" if n_pos else (f"{n_neg}/5 below 0" if n_neg else "0/5")
        if n_pos and n_neg:
            txt = f"{n_pos}/5 above, {n_neg}/5 below"
        ax.text(14.6, y, txt, va="center", ha="left", fontsize=6.8, color=INK)
        y -= 1.0
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(yy, [r[0] for r in rows])
    ax.set_xlim(-14.5, 14.3)
    ax.set_xlabel("difference in test balanced accuracy, percentage points (same split, classifier trained on "
                  "synthetic curves only)")
    ax.text(-14.2, yy[0] + 0.55, "vs. the real-only classifier", fontsize=7, color=INK, style="italic")
    ax.text(-14.2, yy[4] + 0.55, "family vs. family", fontsize=7, color=INK, style="italic")
    ax.grid(axis="x", color="#e6e9ee", lw=0.6, zorder=0)
    ax.legend(handles=[Line2D([], [], marker="o", color=INK, mfc=INK, ms=3.2, lw=1, label="one split: 95% CI excludes 0"),
                       Line2D([], [], marker="o", color=INK, mfc="white", ms=3.2, lw=1, alpha=0.6,
                              label="one split: 95% CI includes 0")],
              loc="lower left", frameon=True, framealpha=0.95, edgecolor="#d0d5dc", fontsize=6.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out / "fig_forest.png")
    plt.close(fig)


def fig_pelsvae(out):
    runs = VAE / "runs"
    jobs = [("tcn", "cVAE (TCN), seed 13", "#1f77b4", "-"), ("tcn_s14", "cVAE (TCN), seed 14", "#1f77b4", "--"),
            ("tcn_phys", "cVAE-P, seed 13", "#9467bd", "-"), ("tcn_phys_s14", "cVAE-P, seed 14", "#9467bd", "--"),
            ("gru", "GRU, seed 13", "#8c564b", "-")]
    paper = {"tcn": 0.011, "tcn_phys": 0.010, "gru": 0.014}
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.5), gridspec_kw=dict(width_ratios=[1.35, 1, 1.15]))
    ax = axes[0]
    for job, lab, c, ls in jobs:
        h = json.load(open(runs / job / "history.json"))
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["val_loss"] for r in h], ls, color=c, lw=1.0, label=lab)
        if job in paper:
            ax.plot(100, paper[job], marker="*", ms=8, color=c, mec="black", mew=0.4, zorder=5)
    h = json.load(open(runs / "tcn" / "history.json"))
    ax2 = ax.twinx()
    ax2.fill_between([r["epoch"] for r in h], [r["lr"] * 1e3 for r in h], color="#e9edf2", lw=0, zorder=0)
    ax2.set_ylim(0, 4.5); ax2.set_yticks([]); ax2.spines["right"].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.text(97, 0.0045, "learning rate\n(grey, restarts at 100)", fontsize=5.8, color="#7d8794", ha="right")
    ax.axvline(100, color="#b0b8c2", lw=0.6, ls=":")
    ax.set_ylim(0.004, 0.036)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation loss")
    ax.set_title("a. Validation loss (stars: paper @ 100)", loc="left")
    ax.legend(fontsize=5.8, loc="upper left", bbox_to_anchor=(0.11, 1.0), frameon=False, handlelength=2.2,
              borderaxespad=0.1)

    ax = axes[1]
    z = np.load(runs / "tcn" / "latent.npz")["mu"]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(z), 4000, replace=False)
    prior = rng.standard_normal((4000, 2))
    ax.scatter(prior[:, 0], prior[:, 1], s=1.0, c="#c3cad3", lw=0, label="prior N(0, I)")
    ax.scatter(z[pick, 0], z[pick, 1], s=1.0, c="#1f77b4", lw=0, alpha=0.6, label="codes of training curves")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    ax.set_xlabel("$z_0$"); ax.set_ylabel("$z_1$", labelpad=0)
    lvp = pd.read_csv(VAE / "tables" / "latent_vs_prior.csv", index_col=0)
    ax.set_title("b. Latent codes vs prior (seed 13)", loc="left")
    ax.text(-3.8, -3.85, f"{lvp.loc['tcn'].iloc[-1]:.0f}% of prior draws fall\noutside the codes' 99% region",
            fontsize=5.8, color=INK, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.legend(fontsize=5.8, loc="upper left", frameon=False, markerscale=4, handletextpad=0.2,
              borderaxespad=0.1)

    ax = axes[2]
    t3 = pd.read_csv(VAE / "tables" / "table3_regression.csv")
    t3 = t3[t3.model == "tcn"].set_index("regressor").loc[["Linear", "RF", "MLP"]]
    x = np.arange(3)
    w = 0.26
    ax.bar(x - w, t3["paper (Table 3)"], w, color="#c9d3de", edgecolor="#8796a8", lw=0.5, label="paper")
    ax.bar(x, t3["train RMSE (in-sample)"], w, color="#5b8fc7", label="ours, in-sample")
    ax.bar(x + w, t3["val RMSE (new stars)"], w, color=INK, label="ours, new stars")
    ax.axhline(t3["predict the mean"].iloc[0], color="#d1495b", lw=0.9, ls="--")
    ax.text(2.45, t3["predict the mean"].iloc[0] + 0.02, "predict the mean code", color="#d1495b",
            fontsize=5.8, ha="right", va="bottom")
    for i, v in enumerate(t3["paper (Table 3)"].values):
        ax.text(i - w, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=5.6)
    ax.set_xticks(x, ["linear", "random\nforest", "MLP"])
    ax.set_ylim(0, 1.42); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("RMSE of the latent code")
    ax.set_title("c. Physical params $\\to$ latent", loc="left")
    ax.legend(fontsize=5.8, loc="upper left", frameon=False, ncol=1, borderaxespad=0.1)
    fig.tight_layout(w_pad=0.9)
    fig.savefig(out / "fig_pelsvae.png")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "report_final/figures")
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_imbalance, fig_per_class, fig_forest, fig_pelsvae):
        f(out)
        print("wrote", f.__name__[4:])
