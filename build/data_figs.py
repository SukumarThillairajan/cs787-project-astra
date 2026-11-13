"""Report figures about the data (2026-11-12): one example star per class, the pre-processing steps on one star,
and the exact tensors the generators and the classifier receive.

Usage (project root; astra imports torch through common.py, so use the venv):
  .venv/Scripts/python.exe build/data_figs.py data/offline_inputs/catalina_raw data/offline_inputs/gan/catalina_L100.npz report/figures
Writes fig_classes.png, fig_preprocess.png, fig_input.png. Reading the raw photometry takes about 2 min once;
the candidate stars are cached in data/offline_inputs/fig_cache.npz (git-ignored).

Example stars are chosen by a rule, not by hand: per class, the 40 stars whose amplitude is closest to the class
median; among those with at least 150 raw points, the one whose folded curve is best described by a 25-bin phase
median (a clear example of a typical star). The pre-processing example is the clearest CEP candidate from which
the outlier filter removes at least 2 points (so that step is visible).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra import catalina as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
                     "ytick.labelsize": 6.8, "legend.fontsize": 6.8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 220})
INK, GREY, KEEP, DROP = "#22303f", "#b9c2cc", "#1f6fb2", "#d1495b"
NAMES = {"EBSD/D": "Contact eclipsing binary (EBSD/D)", "RRL": "RR Lyrae (RRL)",
         "EBC": "Detached eclipsing binary (EBC)", "LPV": "Long-period variable (LPV)",
         "DSCT": "Delta Scuti (DSCT)", "CEP": "Cepheid (CEP)"}
KIND = {"EBSD/D": "two touching stars eclipse each other", "RRL": "old pulsating star",
        "EBC": "separated stars, sharp eclipses", "LPV": "pulsating red giant",
        "DSCT": "fast, small pulsations", "CEP": "pulsating yellow giant"}


def r2_binned(ph, m, nb=25):
    b = np.minimum((ph * nb).astype(int), nb - 1)
    med = np.array([np.median(m[b == k]) if np.any(b == k) else np.nan for k in range(nb)])
    return 1 - np.nanvar(m - med[b]) / np.var(m)


def folded(curve, period):
    t, m, e = curve
    ph = C.fold(t, period)
    o = np.argsort(ph, kind="stable")
    return ph[o], m[o]


def candidates(raw, z, cache):
    """{class: [row index, ...]} and {row index: (t, m, e)} for the 40 amplitude-typical stars of each class."""
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["cand"].item(), d["curves"].item()
    y, amp, numid = z["y"], z["amp"], z["numid"]
    cand = {}
    for c in range(6):
        idx = np.flatnonzero(y == c)
        la = np.log10(amp[idx])
        cand[c] = [int(i) for i in idx[np.argsort(np.abs(la - np.median(la)))[:40]]]
    want = np.concatenate([np.array(v) for v in cand.values()])
    phot = C.read_photometry(raw / "AllVar.phot.gz", numid[want])
    curves = {int(i): phot[numid[i]] for i in want if numid[i] in phot}
    np.savez(cache, cand=np.array(cand, dtype=object), curves=np.array(curves, dtype=object))
    return cand, curves


def best(z, cand, curves, c, min_removed=0):
    top = (-1.0, None)
    for i in cand[c]:
        if i not in curves or len(curves[i][1]) < 150:
            continue
        ph, m = folded(curves[i], z["period"][i])
        if (~C.zscore_keep(m)).sum() < min_removed:
            continue
        r2 = r2_binned(ph, m)
        if r2 > top[0]:
            top = (r2, i)
    return top[1]


def star_icon(ax, cls):
    """A schematic of the star type (not to scale) on a small 'sky' panel."""
    ax.set_facecolor("#0f1b2b")
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    rng = np.random.default_rng(len(cls) + 7)
    ax.scatter(rng.uniform(-1.25, 1.25, 16), rng.uniform(-1, 1, 16), s=0.5, c="white", alpha=0.6, lw=0)
    ring = dict(fill=False, ls=(0, (2, 2)), lw=0.6, ec="#e9eef5", alpha=0.8)
    if cls == "EBSD/D":
        ax.add_patch(patches.Ellipse((0, 0), 1.7, 0.8, fill=False, ls=":", lw=0.6, ec="#8fa3bb"))
        ax.add_patch(patches.Ellipse((-0.24, 0), 0.66, 0.56, fc="#a9c8ff", ec="none"))
        ax.add_patch(patches.Ellipse((0.3, 0), 0.54, 0.46, fc="#ffcf7a", ec="none"))
    elif cls == "EBC":
        ax.add_patch(patches.Ellipse((0, 0), 2.0, 0.66, fill=False, ls=":", lw=0.6, ec="#8fa3bb"))
        ax.add_patch(patches.Circle((-0.62, 0.0), 0.32, fc="#cfe0ff", ec="none"))
        ax.add_patch(patches.Circle((0.78, 0.06), 0.17, fc="#ff9a5a", ec="none"))
    else:
        r, col = {"RRL": (0.34, "#fff1b8"), "LPV": (0.56, "#ff7f50"), "DSCT": (0.22, "#eef4ff"),
                  "CEP": (0.44, "#ffd76a")}[cls]
        ax.add_patch(patches.Circle((0, 0), r, fc=col, ec="none"))
        for k in (1.25, 1.55):
            ax.add_patch(patches.Circle((0, 0), r * k, **ring))


def fig_classes(z, pick, curves, out):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    n_cls = np.bincount(z["y"], minlength=6)
    for c, ax in enumerate(axes.ravel()):
        cls = C.CLASSES[c]
        i = pick[c]
        ph, m = folded(curves[i], z["period"][i])
        k = C.zscore_keep(m)
        ax.scatter(np.r_[ph[k], ph[k] + 1], np.r_[m[k], m[k]], s=2.0, c=INK, lw=0, alpha=0.75)
        ax.invert_yaxis()
        ax.set_xlim(0, 2)
        ax.set_xlabel("phase (two cycles shown)")
        if c % 3 == 0:
            ax.set_ylabel("magnitude (brighter up)")
        p = z["period"][i]
        ptxt = f"{p:.0f} d" if p > 20 else (f"{p:.2f} d" if p > 0.2 else f"{p * 24:.1f} h")
        ax.set_title(f"{NAMES[cls]}\n{KIND[cls]}\nP = {ptxt};  {n_cls[c]:,} curves in the data", loc="left",
                     fontsize=7.4)
        ins = ax.inset_axes([0.74, 1.04, 0.26, 0.36])
        star_icon(ins, cls)
    fig.tight_layout(h_pad=1.4, w_pad=0.8)
    fig.savefig(out / "fig_classes.png")
    plt.close(fig)


def fig_preprocess(z, i, curve, out):
    t, m, e = curve
    P = z["period"][i]
    ph = C.fold(t, P)
    o = np.argsort(ph, kind="stable")
    phs, ms = ph[o], m[o]
    k = C.zscore_keep(ms)
    # the 100 points the benchmark actually uses, matched back to the raw points by phase
    zp, zm, A = z["phase"][i].astype(np.float64), z["mag"][i].astype(np.float64), float(z["amp"][i])
    j = np.searchsorted(phs, zp).clip(1, len(phs) - 1)
    j = np.where(np.abs(phs[j - 1] - zp) < np.abs(phs[j] - zp), j - 1, j)
    assert np.abs(phs[j] - zp).max() < 1e-5, "could not match the kept points to the raw ones"
    sel = np.zeros(len(phs), bool)
    sel[j] = True
    fig, axes = plt.subplots(1, 5, figsize=(7.6, 2.25), gridspec_kw=dict(width_ratios=[1.3, 1, 1, 1, 1]))
    ylim = (ms.max() + 0.06, ms.min() - 0.06)          # one magnitude scale for panels 1-4, brighter up
    ax = axes[0]
    ax.scatter(t - t.min(), m, s=1.5, c=INK, lw=0)
    ax.set_ylim(*ylim); ax.set_xlabel("days since first point"); ax.set_ylabel("magnitude (brighter up)")
    ax.set_title(f"1. Raw: {len(m)} points\nover {(t.max() - t.min()) / 365.25:.1f} years", loc="left")
    ax = axes[1]
    ax.scatter(phs, ms, s=1.5, c=INK, lw=0)
    ax.set_ylim(*ylim); ax.set_xlabel("phase"); ax.set_yticklabels([])
    ax.set_title(f"2. Fold at the period\nP = {P:.3f} d", loc="left")
    ax = axes[2]
    ax.scatter(phs[k], ms[k], s=1.5, c=INK, lw=0)
    ax.scatter(phs[~k], ms[~k], s=18, marker="x", c=DROP, lw=1.0)
    ax.set_ylim(*ylim); ax.set_xlabel("phase"); ax.set_yticklabels([])
    ax.set_title(f"3. Remove outliers\n(z-score; {(~k).sum()} red x)", loc="left")
    ax = axes[3]
    ax.scatter(phs[k & ~sel], ms[k & ~sel], s=1.5, c=GREY, lw=0)
    ax.scatter(phs[sel], ms[sel], s=4, c=KEEP, lw=0)
    ax.set_ylim(*ylim); ax.set_xlabel("phase"); ax.set_yticklabels([])
    ax.set_title("4. Keep 100 random\npoints (blue)", loc="left")
    ax = axes[4]
    ax.axhspan(-1, 1, color="#eef3f8", lw=0)
    ax.scatter(zp, zm / A, s=4, c=KEEP, lw=0)
    ax.invert_yaxis(); ax.set_xlabel("phase"); ax.set_ylabel("u = (m - median) / A", labelpad=1)
    ax.set_title(f"5. Centre, scale\n(A = {A:.2f} mag)", loc="left")
    for a in axes[1:]:
        a.set_xlim(-0.03, 1.03); a.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "fig_preprocess.png", dpi=300)
    plt.close(fig)


def fig_input(z, i, out):
    zp, zm, A = z["phase"][i].astype(np.float64), z["mag"][i].astype(np.float64), float(z["amp"][i])
    cls = C.CLASSES[int(z["y"][i])]
    u = zm / A
    L = len(u)
    x0 = np.random.default_rng(3).standard_normal(L)
    fig = plt.figure(figsize=(7.4, 5.9))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.25, 0.2, 0.42], hspace=0.92)
    top = gs[0].subgridspec(1, 5, width_ratios=[1.5, 1, 1, 1, 1], wspace=0.35)
    ax = fig.add_subplot(top[0])
    ax.scatter(zp, u, s=5, c=KEEP, lw=0)
    ax.plot(zp, np.full(L, 1.25), "|", c="#7d8a99", ms=4, mew=0.6)
    ax.set_ylim(1.4, -1.15); ax.set_xlabel("phase (ticks: where the 100 points are)"); ax.set_ylabel("u")
    ax.set_title(f"a. One training curve ({cls})", loc="left")
    mini = []
    for n, tt in enumerate([0.0, 0.4, 0.7, 1.0]):
        ax = fig.add_subplot(top[n + 1])
        mini.append(ax)
        ax.scatter(zp, (1 - tt) * x0 + tt * u, s=2.5, c=KEEP if tt == 1 else "#7f93a8", lw=0)
        ax.set_ylim(3, -3); ax.set_xticks([0, 1]); ax.set_xlabel("phase")
        if n:
            ax.set_yticklabels([])
        ax.set_title(f"t = {tt:g}" + ("  (noise)" if tt == 0 else "  (data)" if tt == 1 else ""), fontsize=7)
    pos = mini[0].get_position()
    fig.text(pos.x0, pos.y1 + 0.045, r"b. Flow matching: noise $\epsilon$ moves to the curve along straight "
             r"lines, $x_t = (1-t)\,\epsilon + t\,u$", fontsize=8, va="bottom")
    tt = 0.5
    rows, labels = [(1 - tt) * x0 + tt * u], [r"$x_t$ (here t = 0.5)"]
    for kk in range(1, 5):
        rows += [np.sin(2 * np.pi * kk * zp), np.cos(2 * np.pi * kk * zp)]
        labels += [rf"sin $2\pi{kk}\phi$", rf"cos $2\pi{kk}\phi$"]
    M = np.vstack(rows)
    ax = fig.add_subplot(gs[1])
    vmax = np.abs(M).max()
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest",
                   extent=(0.5, L + 0.5, len(rows) - 0.5, -0.5))
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("position in the sequence (points sorted by phase)")
    ax.set_title("c. Input of the DDPM / flow-matching U-Net: 9 channels x 100 positions "
                 "(convolutions wrap around, as phase is periodic)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[2])
    ax.axis("off")
    ax.text(0.0, 1.0, "Conditioning, one vector added inside every residual block:   e = "
            "Emb(class)  +  MLP(standardised log10 A)  +  MLP(sinusoidal(1000 t))\n"
            f"here: class = {cls}, A = {A:.2f} mag.  Training drops the class sometimes "
            "(a 7th 'no class' row), which allows classifier-free guidance.\n"
            "The GAN and the VAE get the same condition: the class, the 100 phases and the amplitude of a real star.",
            transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(fc="#f4f6f9", ec="#c9d1db", lw=0.6, boxstyle="round,pad=0.4"))
    ax = fig.add_subplot(gs[3])
    C2 = np.vstack([zm / np.abs(zm).max(), zp * 2 - 1])
    ax.imshow(C2, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
              extent=(0.5, L + 0.5, 1.5, -0.5))
    ax.set_yticks([0, 1]); ax.set_yticklabels(["magnitude", "phase"])
    ax.set_xlabel("position in the sequence")
    ax.set_title("d. Input of the classifier (ensemble of 5 CNNs): 2 channels x 100 positions; "
                 "magnitude = real, or generated u x A", loc="left")
    fig.savefig(out / "fig_input.png", bbox_inches="tight")
    plt.close(fig)


def main(raw, npz, out):
    raw, out = Path(raw), Path(out)
    z = dict(np.load(npz))
    cand, curves = candidates(raw, z, Path("data/offline_inputs/fig_cache.npz"))
    pick = {c: best(z, cand, curves, c) for c in range(6)}
    pre = best(z, cand, curves, C.CLASSES.index("CEP"), min_removed=2)
    for c in range(6):
        i = pick[c]
        print(C.CLASSES[c], "numid", int(z["numid"][i]), "P", round(float(z["period"][i]), 4),
              "A", round(float(z["amp"][i]), 3), "raw points", len(curves[i][1]))
    print("pre-processing example: CEP numid", int(z["numid"][pre]), "raw points", len(curves[pre][1]))
    fig_classes(z, pick, curves, out)
    fig_preprocess(z, pre, curves[pre], out)
    fig_input(z, pre, out)
    print("wrote fig_classes.png, fig_preprocess.png, fig_input.png to", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
