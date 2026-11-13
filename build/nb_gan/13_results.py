# ---- Table 5 and figures from everything finished so far ----
from astra.metrics import welch_p

PAPER = {"catalina": {"A": 73.44, "B": 74.06, "C": 73.64, "D": 73.82, "E": 75.97, "F": 76.28, "G": 76.86, "H": 76.97},
         "ztf": {"A": 72.61, "B": 72.69, "C": 72.45, "D": 72.53, "E": 74.17, "F": 73.79, "G": 74.37, "H": 74.94}}[DATASET]
PAPER_SD = {"catalina": {"A": 1.22, "B": 1.04, "C": 1.79, "D": 1.50, "E": 0.94, "F": 0.74, "G": 1.09, "H": 0.79},
            "ztf": {"A": 0.69, "B": 0.99, "C": 0.70, "D": 0.69, "E": 0.62, "F": 0.50, "G": 0.51, "H": 0.44}}[DATASET]
load = lambda p: json.loads(p.read_text()) if p.exists() else None
trtr = load(RUNS / "trtr" / "trtr.json") or {}
rows = {}                                     # row -> (label, result dict)
for key, name in (("A", "A_oversampling"), ("B", "B_soft_time_warping"), ("C", "C_soft_mag_warping"),
                  ("D", "D_soft_mixed_warping")):
    if name in trtr:
        rows[key] = (f"TRTR {name[2:].replace('_', ' ')}", trtr[name])
gan_jobs = sorted(p.parent.name for p in RUNS.glob("gan_*/config.json"))
for job in gan_jobs:
    fin = load(RUNS / job / "final.json") or {}
    base = {"gan_u_s0": ("E", "F"), "gan_gamma025_s0": ("G", "H")}.get(job)
    for i, crit in enumerate(("val_acc", "g_score")):
        if crit in fin:
            key = base[i] if base else f"{job}:{crit}"
            rows[key] = (f"TSTR {job} [{crit}, ckpt {fin[crit]['checkpoint'] // 1000}k]", fin[crit])

rows = dict(sorted(rows.items()))            # A..H first, then the extra seeds
tab = []
for key, (label, r) in rows.items():
    acc = 100 * np.array(r["test_acc"])
    tab.append({"row": key, "method": label, "ours_acc": f"{acc.mean():.2f} ± {acc.std():.2f}",
                "paper_acc": f"{PAPER[key]:.2f} ± {PAPER_SD[key]:.2f}" if key in PAPER else "",
                "diff_vs_paper": round(acc.mean() - PAPER[key], 2) if key in PAPER else None,
                "p_vs_A": round(welch_p(acc, 100 * np.array(rows["A"][1]["test_acc"])), 4) if "A" in rows and key != "A" else None,
                "macro_F1": f"{100 * np.mean(r['test_f1']):.2f}", "runs": len(acc),
                "epochs(best, mean)": round(float(np.mean(r["best_epoch"])), 1)})
tab = pd.DataFrame(tab)
tab.to_csv(TABLES / "table5_ours_vs_paper.csv", index=False)
print(f"Table 5 ({DATASET}) — test accuracy %, mean ± std over runs")
print(tab.to_string(index=False) if len(tab) else "(no finished rows yet)")
if len(rows) > 1:
    keys = list(rows)
    pv = pd.DataFrame([[welch_p(rows[a][1]["test_acc"], rows[b][1]["test_acc"]) if a != b else np.nan for b in keys]
                       for a in keys], index=keys, columns=keys).round(4)
    pv.to_csv(TABLES / "welch_pvalues.csv")
    print("\nWelch's t-test p-values between rows:\n", pv.to_string())
    rec = pd.DataFrame({k: 100 * np.mean(r["test_recall_per_class"], 0) for k, (_, r) in rows.items()},
                       index=CLASSES).round(1)
    rec.to_csv(TABLES / "per_class_recall.csv")
    print("\nPer-class test recall % (classes from most to least common):\n", rec.to_string())

# ---- curves over GAN training (Fig. 3, 4, 7 style) ----
M = {j: load(RUNS / j / "metrics.json") for j in gan_jobs}
M = {j: m for j, m in M.items() if m}
if M:
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    for j, m in M.items():
        it = np.array([a["it"] for a in m]) / 1000
        va = np.array([a["val_acc"] for a in m]); vs = np.array([np.std(a["val_acc_runs"]) for a in m])
        ax[0].plot(it, 100 * va, label=j); ax[0].fill_between(it, 100 * (va - vs), 100 * (va + vs), alpha=.2)
        ax[1].plot(it, [a["fid"] for a in m], label=j)
        if "g_score" in m[0]:
            ax[2].plot(it, [a["g_score"] for a in m], label=j)
    for a, t in zip(ax, ("validation TSTR accuracy %", "FID (lower = closer)", "G-score (after training)")):
        a.set_xlabel("generator iterations (k)"); a.set_title(t); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "gan_training_curves.png", dpi=110); plt.show()

    fig, ax = plt.subplots(1, len(M), figsize=(5 * len(M), 4), squeeze=False)
    for a, (j, m) in zip(ax[0], M.items()):
        D = np.array([x["density"] for x in m]).mean(1); R = np.array([x["recall"] for x in m]).mean(1)
        sc = a.scatter(R, D, c=[x["it"] / 1000 for x in m], cmap="viridis")
        fin = load(RUNS / j / "final.json") or {}
        for crit, mk in (("val_acc", "s"), ("g_score", "*")):
            if crit in fin:
                k = [x["it"] for x in m].index(fin[crit]["checkpoint"])
                a.scatter(R[k], D[k], marker=mk, s=250, facecolors="none", edgecolors="r", label=crit)
        a.set_xlabel("macro recall (diversity)"); a.set_ylabel("macro density (fidelity)"); a.set_title(j)
        a.legend(); plt.colorbar(sc, ax=a, label="iterations (k)")
    fig.tight_layout(); fig.savefig(FIGS / "density_recall_plane.png", dpi=110); plt.show()

    j0 = "gan_gamma025_s0" if "gan_gamma025_s0" in M else next(iter(M))
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    it = np.array([a["it"] for a in M[j0]]) / 1000
    for c, name in enumerate(CLASSES):
        ax[0].plot(it, [a["density"][c] for a in M[j0]], label=name)
        ax[1].plot(it, [a["recall"][c] for a in M[j0]], label=name)
    ax[0].set_title(f"{j0}: density per class"); ax[1].set_title(f"{j0}: recall per class")
    for a in ax:
        a.set_xlabel("generator iterations (k)"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "per_class_density_recall.png", dpi=110); plt.show()

# ---- real vs generated with the same conditions (Fig. 9 style) ----
for j in gan_jobs:
    s = RUNS / j / "samples_g_score.npz"
    if not s.exists():
        continue
    S = np.load(s)
    fig, ax = plt.subplots(2, len(CLASSES), figsize=(18, 5), sharex=True)
    for c, name in enumerate(CLASSES):
        k = np.flatnonzero(Z["y"][S["idx"]] == c)[:2]
        for r, i in enumerate(k):
            ax[r, c].plot(Z["phase"][S["idx"][i]], Z["mag"][S["idx"][i]], ".", ms=3, label="real")
            ax[r, c].plot(Z["phase"][S["idx"][i]], S["x"][i], ".", ms=3, label="generated")
            ax[r, c].invert_yaxis(); ax[r, c].set_title(name, fontsize=9)
    ax[0, 0].legend(fontsize=7); fig.suptitle(f"{j}, G-score checkpoint: real vs generated, same condition")
    fig.tight_layout(); fig.savefig(FIGS / f"samples_{j}.png", dpi=110); plt.show()

# ---- over/under-fitting of the TRTR classifiers ----
if trtr:
    fig, ax = plt.subplots(1, len(trtr), figsize=(4.5 * len(trtr), 3.2), squeeze=False)
    for a, (name, r) in zip(ax[0], trtr.items()):
        a.plot(r["curves"]["epoch"], r["curves"]["train_loss"], label="train")
        a.plot(r["curves"]["epoch"], r["curves"]["val_loss"], label="val")
        a.set_title(name, fontsize=9); a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(FIGS / "trtr_loss_curves.png", dpi=110); plt.show()
