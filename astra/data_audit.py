"""Dataset audit for the ASTRA project: tables and figures that describe the light-curve datasets and justify
the choices of the benchmark (split over stars, balanced accuracy, the rare tier, the amplitude-normalised target).

Everything here runs on a CPU in a few minutes and trains nothing except one small classifier per dataset
(for the confusion matrix and the feature map). Z is a pre-processed file (catalina_L100.npz or ztf_L40.npz)
opened with numpy.load; its `split` column is the split of Garcia-Jara et al. (2022), kept for comparison.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import bench_data, classifier

TRAIN, VAL, TEST = 0, 1, 2
COLORS = plt.get_cmap("tab10").colors


def classes_of(Z):
    return [str(c) for c in Z["classes"]]


def object_ids(Z):
    return Z["numid"] if "numid" in Z.files else Z["oid"]


def class_table(Z, seed=0):
    """Curves, object ids and stars per class, and the sizes of our split over stars and of the paper's split."""
    y, names = Z["y"], classes_of(Z)
    star = bench_data.star_ids(y, Z["period"])
    ours = bench_data.star_split(y, star, seed)
    rows = []
    for c, name in enumerate(names):
        m = y == c
        rows.append({"class": name, "curves": int(m.sum()), "object ids": len(np.unique(object_ids(Z)[m])),
                     "stars": len(np.unique(star[m])),
                     "train": int((m & (ours == TRAIN)).sum()), "val": int((m & (ours == VAL)).sum()),
                     "test": int((m & (ours == TEST)).sum()),
                     "paper train": int((m & (Z["split"] == TRAIN)).sum()), "paper test": int((m & (Z["split"] == TEST)).sum())})
    t = pd.DataFrame(rows)
    t["share of curves %"] = (100 * t.curves / t.curves.sum()).round(2)
    t["rare tier"] = t.train < bench_data.RARE_BELOW
    return t


def plot_imbalance(tables, path):
    """Curves per class on a log scale, one panel per dataset; the rare tier is hatched."""
    fig, ax = plt.subplots(1, len(tables), figsize=(6 * len(tables), 3.6), squeeze=False)
    for a, (name, t) in zip(ax[0], tables.items()):
        bars = a.bar(t["class"], t.curves, color=[COLORS[i] for i in range(len(t))])
        for b, rare, n in zip(bars, t["rare tier"], t.curves):
            if rare:
                b.set_hatch("//")
            a.text(b.get_x() + b.get_width() / 2, n * 1.08, f"{n:,}", ha="center", fontsize=8)
        a.set_yscale("log"); a.set_ylabel("light curves (log scale)")
        a.set_title(f"{name}: {t.curves.sum():,} curves, largest / smallest class = {t.curves.max() / t.curves.min():.0f}x "
                    "(hatched = rare tier)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.show()


def plot_gallery(Z, name, path, per_class=3, seed=1):
    """Folded light curves, a few per class, as the models see them (magnitudes grow downwards)."""
    names, rng = classes_of(Z), np.random.default_rng(seed)
    fig, ax = plt.subplots(per_class, len(names), figsize=(2.9 * len(names), 2.1 * per_class), sharex=True, squeeze=False)
    for c, cname in enumerate(names):
        for r, i in enumerate(rng.choice(np.flatnonzero(Z["y"] == c), per_class, replace=False)):
            err = np.where((Z["err"][i] > 0) & (Z["err"][i] < 1), Z["err"][i], 0)
            ax[r, c].errorbar(Z["phase"][i], Z["mag"][i], yerr=err, fmt=".", ms=3, lw=0.5, color=COLORS[c], ecolor="0.75")
            ax[r, c].invert_yaxis()
            ax[r, c].set_title(f"{cname} · P = {Z['period'][i]:.3g} d · A = {Z['amp'][i]:.2f}", fontsize=8)
    fig.supxlabel("phase"); fig.supylabel("centred magnitude")
    fig.suptitle(f"{name}: folded light curves ({Z['mag'].shape[1]} points each)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.show()


def plot_period_amplitude(Z, name, path, max_per_class=3000, seed=0):
    """Period against amplitude: what the conditions (class, amplitude) and the period say about a star."""
    names, rng = classes_of(Z), np.random.default_rng(seed)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    order = np.argsort(-np.bincount(Z["y"], minlength=len(names)))          # common classes first, rare ones on top
    for c in order:
        i = np.flatnonzero(Z["y"] == c)
        i = rng.choice(i, min(len(i), max_per_class), replace=False)
        ax[0].scatter(Z["period"][i], Z["amp"][i], s=4, alpha=0.35, color=COLORS[c], label=names[c])
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("period (days)"); ax[0].set_ylabel("amplitude (mag)")
    leg = ax[0].legend(fontsize=8, markerscale=3)
    for h in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
        h.set_alpha(1)
    for a, key, label in ((ax[1], "period", "log10 period (days)"), (ax[2], "amp", "log10 amplitude (mag)")):
        parts = a.violinplot([np.log10(Z[key][Z["y"] == c]) for c in range(len(names))], showmedians=True, widths=0.9)
        for body, c in zip(parts["bodies"], range(len(names))):
            body.set_facecolor(COLORS[c]); body.set_alpha(0.6)
        a.set_xticks(range(1, len(names) + 1)); a.set_xticklabels(names, fontsize=8); a.set_ylabel(label)
    fig.suptitle(f"{name}: period and amplitude per class", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.show()
    return pd.DataFrame({"class": names,
                         "median period (d)": [float(np.median(Z["period"][Z["y"] == c])) for c in range(len(names))],
                         "median amplitude (mag)": [float(np.median(Z["amp"][Z["y"] == c])) for c in range(len(names))]}).round(3)


def largest_gap(phase):
    """Largest empty stretch of the phase circle for every curve (phases sorted in [0, 1))."""
    gaps = np.diff(phase, axis=1)
    wrap = 1 - phase[:, -1] + phase[:, 0]
    return np.maximum(gaps.max(1), wrap)


def plot_sampling(Z, name, path, seed=0):
    """How irregular the sampling is: the largest phase gap of every curve, against what L uniformly random
    phases would give."""
    names, L = classes_of(Z), Z["phase"].shape[1]
    gap = largest_gap(Z["phase"])
    ref = np.median(largest_gap(np.sort(np.random.default_rng(seed).random((20_000, L)), 1)))
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    bins = np.linspace(0, 0.6, 61)
    for c, cname in enumerate(names):
        ax[0].hist(gap[Z["y"] == c], bins=bins, histtype="step", density=True, color=COLORS[c], label=cname)
    ax[0].axvline(ref, color="k", ls="--", lw=1, label=f"{L} random phases (median)")
    ax[0].set_xlabel("largest phase gap of a curve"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=7)
    worst = np.argsort(gap)[-3:]
    for k, i in enumerate(worst):
        ax[1].plot(Z["phase"][i], Z["mag"][i] - Z["mag"][i].mean() + 1.2 * k * Z["amp"][i].clip(0.1), ".", ms=4,
                   label=f"{names[Z['y'][i]]}, gap {gap[i]:.2f}")
    ax[1].invert_yaxis(); ax[1].set_xlabel("phase"); ax[1].set_yticks([]); ax[1].legend(fontsize=7)
    ax[1].set_title("the three curves with the largest gaps (shifted vertically)", fontsize=9)
    fig.suptitle(f"{name}: irregular sampling in phase", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.show()
    t = pd.DataFrame({"class": names,
                      "median largest gap": [float(np.median(gap[Z["y"] == c])) for c in range(len(names))],
                      "curves with a gap > 0.2 (%)": [100 * float(np.mean(gap[Z["y"] == c] > 0.2)) for c in range(len(names))]})
    t["random phases would give"] = ref
    return t.round(3)


def plot_noise(Z, name, path):
    """Photometric error relative to the amplitude: how much of a curve is signal."""
    names = classes_of(Z)
    err = np.where((Z["err"] > 0) & (Z["err"] < 1), Z["err"], np.nan)            # flagged errors (>= 1 mag) ignored
    med = np.nanmedian(err, 1)
    rel = med / Z["amp"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for a, v, label in ((ax[0], med, "median error of a curve (mag)"), (ax[1], rel, "median error / amplitude")):
        data = [np.log10(v[(Z["y"] == c) & np.isfinite(v)]) for c in range(len(names))]
        parts = a.violinplot(data, showmedians=True, widths=0.9)
        for body, c in zip(parts["bodies"], range(len(names))):
            body.set_facecolor(COLORS[c]); body.set_alpha(0.6)
        a.set_xticks(range(1, len(names) + 1)); a.set_xticklabels(names, fontsize=8); a.set_ylabel("log10 " + label)
    fig.suptitle(f"{name}: photometric noise per class", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.show()
    return pd.DataFrame({"class": names,
                         "median error (mag)": [float(np.nanmedian(med[Z["y"] == c])) for c in range(len(names))],
                         "median error / amplitude": [float(np.nanmedian(rel[Z["y"] == c])) for c in range(len(names))]}).round(3)


def leakage_table(Z, seed=0):
    """Do curves of one star sit on both sides of a split? A star = all curves with the same class and period
    (two bands, or two object ids of one catalogue source). Paper's split: the one stored in the file."""
    y, names = Z["y"], classes_of(Z)
    star = bench_data.star_ids(y, Z["period"])
    ours = bench_data.star_split(y, star, seed)
    per_star = np.bincount(star)
    rows = []
    for c, name in enumerate(names):
        m = y == c
        row = {"class": name, "curves": int(m.sum()), "stars": len(np.unique(star[m])),
               "curves in stars with more than one curve (%)": 100 * float(np.mean(per_star[star[m]] > 1))}
        for label, split in (("paper's split", Z["split"]), ("our split", ours)):
            train_stars = np.unique(star[split == TRAIN])
            held = m & (split != TRAIN)
            row[f"{label}: held-out curves with a same-star curve in training (%)"] = \
                100 * float(np.isin(star[held], train_stars).mean()) if held.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).round(1)


def power_table(Z, seed=0, recall=0.75):
    """How finely a test set can measure a class: curves per class, what one curve is worth in recall points,
    and the binomial standard error of a recall of 75%."""
    y, names = Z["y"], classes_of(Z)
    ours = bench_data.star_split(y, bench_data.star_ids(y, Z["period"]), seed)
    rows = []
    for c, name in enumerate(names):
        row = {"class": name}
        for label, split in (("paper", Z["split"]), ("ours", ours)):
            n = int(((y == c) & (split == TEST)).sum())
            row[f"test curves ({label})"] = n
            row[f"one curve = points ({label})"] = 100 / max(n, 1)
            row[f"std. error of a 75% recall ({label})"] = 100 * np.sqrt(recall * (1 - recall) / max(n, 1))
        rows.append(row)
    return pd.DataFrame(rows).round(2)


def confusion_and_map(npz_path, name, path, seed=0, max_epochs=6, n_map=400, device="cpu"):
    """One CNN (the paper's base classifier) trained on the real training curves of our split: its confusion
    matrix on the test stars, and a t-SNE map of its last-conv features for a class-balanced sample of them."""
    from sklearn.manifold import TSNE
    data = bench_data.load(npz_path, seed, device)
    names, N = data["classes"], data["n_classes"]
    tr, va, te = (bench_data.subset(data, data[f"{s}_idx"]) for s in ("train", "val", "test"))
    model, _ = classifier.train_feature_cnn(tr, va, N, max_epochs=max_epochs, seed=seed, device=device, log=lambda *a: None)
    X = classifier.stack_input(te["x"], te["ph"])
    pred = classifier.predict_proba(model, X, amp=False)[0].argmax(1).cpu().numpy()
    yte = data["y_np"][data["test_idx_np"]]
    conf = np.zeros((N, N))
    np.add.at(conf, (yte, pred), 1)
    rec = conf / conf.sum(1, keepdims=True)
    rng = np.random.default_rng(seed)
    pick = np.concatenate([rng.choice(np.flatnonzero(yte == c), min(n_map, int((yte == c).sum())), replace=False)
                           for c in range(N)])
    feats = classifier.extract_features(model, te["x"][torch.from_numpy(pick)], te["ph"][torch.from_numpy(pick)]).cpu().numpy()
    xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(feats)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    im = ax[0].imshow(100 * rec, cmap="Blues", vmin=0, vmax=100)
    for i in range(N):
        for j in range(N):
            ax[0].text(j, i, f"{100 * rec[i, j]:.0f}", ha="center", va="center", fontsize=9,
                       color="w" if rec[i, j] > 0.5 else "k")
    ax[0].set_xticks(range(N)); ax[0].set_xticklabels(names, fontsize=8); ax[0].set_yticks(range(N))
    ax[0].set_yticklabels([f"{n} ({int(conf[i].sum())})" for i, n in enumerate(names)], fontsize=8)
    ax[0].set_xlabel("predicted class"); ax[0].set_ylabel("true class (test curves)")
    ax[0].set_title(f"confusion matrix, % of each true class | balanced accuracy {100 * np.mean(np.diag(rec)):.1f}%, "
                    f"accuracy {100 * np.mean(pred == yte):.1f}%", fontsize=9)
    plt.colorbar(im, ax=ax[0], fraction=0.046)
    for c in np.argsort(-np.bincount(yte[pick], minlength=N)):
        m = yte[pick] == c
        ax[1].scatter(xy[m, 0], xy[m, 1], s=7, alpha=0.6, color=COLORS[c], label=names[c])
    ax[1].legend(fontsize=8, markerscale=2); ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title(f"t-SNE of the classifier's features (up to {n_map} test curves per class)", fontsize=9)
    fig.suptitle(f"{name}: where a classifier trained on real data confuses the classes (one CNN, split over stars)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.show()
    return pd.DataFrame(100 * rec, index=names, columns=names).round(1), \
        {"balanced accuracy %": 100 * float(np.mean(np.diag(rec))), "accuracy %": 100 * float(np.mean(pred == yte))}


def ogle_table(meta, split_seed=13):
    """The PELS-VAE training file: curves against unique stars per class, the share of curves that have the
    three physical parameters, and how many validation stars of their random split have a copy in training."""
    from .pelsvae import split_indices
    tr, va = split_indices(len(meta), split_seed)
    ids, phys = meta.OGLE_id.values, meta[["Period", "teff_val", "abs_Gmag"]].notna().all(1).values
    in_train = set(ids[tr])
    rows = []
    for name in sorted(meta.Type.unique()):
        m = meta.Type.values == name
        stars_va = set(ids[va][m[va]])
        rows.append({"class": name, "curves": int(m.sum()), "stars": len(set(ids[m])),
                     "curves per star": m.sum() / len(set(ids[m])),
                     "curves with period, T_eff and abs. G mag (%)": 100 * float(phys[m].mean()),
                     "validation stars with a copy in training (%)": 100 * len(stars_va & in_train) / max(len(stars_va), 1)})
    return pd.DataFrame(rows).round(1)


def plot_ogle(t, path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.stars, 0.4, label="unique stars"); ax.bar(x + 0.2, t.curves, 0.4, label="curves in the training file")
    for i, r in t.iterrows():
        ax.text(i + 0.2, r.curves * 1.1, f"x{r['curves per star']:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(t["class"], fontsize=8); ax.set_yscale("log"); ax.legend(fontsize=8)
    ax.set_title("PELS-VAE training file (OGLE-III): every class is filled up with augmented copies", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.show()


def pipeline_example(catalog_path, phot_path, Z, path, seed=0):
    """Catalina only: two stars from raw photometry to model input
    (time series -> folded by its period -> outlier filter -> L random points, median-centred)."""
    from . import catalina
    cat = catalina.read_catalog(catalog_path).set_index("numid")
    rng = np.random.default_rng(seed)
    picks = [int(Z["numid"][rng.choice(np.flatnonzero(Z["y"] == c))]) for c in (1, len(Z["classes"]) - 1)]   # an RRL and a CEP
    phot = catalina.read_photometry(phot_path, picks)
    fig, ax = plt.subplots(len(picks), 3, figsize=(15, 3.2 * len(picks)), squeeze=False)
    for r, numid in enumerate(picks):
        t, m, e = phot[numid]
        period = float(cat.loc[numid, "period"])
        ph = catalina.fold(t, period)
        o = np.argsort(ph, kind="stable")
        keep = catalina.zscore_keep(m[o])
        i = int(np.flatnonzero(Z["numid"] == numid)[0])
        ax[r, 0].plot(t - t.min(), m, ".", ms=3, color="0.3"); ax[r, 0].invert_yaxis()
        ax[r, 0].set_xlabel("days since the first observation"); ax[r, 0].set_ylabel("magnitude")
        ax[r, 0].set_title(f"{cat.loc[numid, 'cls']}: {len(t)} observations over {(t.max() - t.min()) / 365.25:.1f} years", fontsize=9)
        ax[r, 1].plot(ph[o][keep], m[o][keep], ".", ms=3, color="C0", label="kept")
        ax[r, 1].plot(ph[o][~keep], m[o][~keep], "x", ms=5, color="C3", label="outliers removed")
        ax[r, 1].invert_yaxis(); ax[r, 1].set_xlabel("phase = (time mod period) / period"); ax[r, 1].legend(fontsize=7)
        ax[r, 1].set_title(f"folded with its period, {period:.4g} days", fontsize=9)
        ax[r, 2].plot(Z["phase"][i], Z["mag"][i], ".", ms=4, color="C1"); ax[r, 2].invert_yaxis()
        ax[r, 2].set_xlabel("phase"); ax[r, 2].set_title(f"model input: {Z['mag'].shape[1]} random points, median-centred", fontsize=9)
    fig.suptitle("Catalina: from raw photometry to the curve a model sees", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.show()
