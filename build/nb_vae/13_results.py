# ---- Figs. 4-7 from the finished jobs ----
load = lambda p: json.loads(p.read_text()) if p.exists() else None
jobs = [j for q in QUEUES.values() for j in q]
H = {j: load(RUNS / j / "history.json") for j in jobs}
H = {j: h for j, h in H.items() if h}
PAPER_FIG4 = {"tcn": 0.011, "tcn_phys": 0.010, "gru": 0.014}          # read off their Fig. 4 at epoch 100
rows = []
for j, h in H.items():
    at = lambda e, k: next((x[k] for x in h if x["epoch"] == e), np.nan)
    rows.append({"job": j, "epochs": len(h), "val loss @100": at(100, "val_loss"), "val loss @150": at(150, "val_loss"),
                 "paper @100 (Fig. 4)": PAPER_FIG4.get(j, np.nan),
                 "min val loss": min(x["val_loss"] for x in h), "at epoch": min(h, key=lambda x: x["val_loss"])["epoch"],
                 "train loss (last)": h[-1]["train_loss"],
                 "val MSE (last)": h[-1]["val_mse"], "val KL/1e4 (last)": h[-1]["val_kl"],
                 "min/epoch": np.mean([x["seconds"] for x in h]) / 60})
fig4 = pd.DataFrame(rows).round(5)
fig4.to_csv(TABLES / "fig4_losses.csv", index=False)
print(fig4.to_string(index=False) if len(fig4) else "(no finished jobs yet)")

if H:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for j, h in H.items():
        e = [x["epoch"] for x in h]
        ls = "--" if "s14" in j else "-"
        ax[0].plot(e, [x["val_loss"] for x in h], ls, label=j)
        ax[1].plot(e, [x["train_loss"] for x in h], ls, label=f"{j} train")
        ax[1].plot(e, [x["val_loss"] for x in h], ":", label=f"{j} val")
    for j, v in PAPER_FIG4.items():
        ax[0].scatter([100], [v], marker="x", s=80, label=f"paper {j} @100")
    ax[0].set_ylim(0, 0.06); ax[0].set_title("validation loss (their Fig. 4)"); ax[0].legend(fontsize=7)
    ax[1].set_yscale("log"); ax[1].set_title("train vs validation (over/under-fitting)"); ax[1].legend(fontsize=6)
    for a in ax:
        a.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(FIGS / "fig4_val_loss.png", dpi=110); plt.show()

# ---- reconstructions (Figs. 5 and 7) ----
classes = sorted(meta.Type.unique())
for j in ("tcn", "tcn_s14", "tcn_phys"):
    p = RUNS / j / "recon.npz"
    if not p.exists():
        continue
    R = np.load(p)
    split = np.load(RUNS / j / "split.npz")
    y = meta.Type.values[split["rows"][R["idx"]]]
    fig, ax = plt.subplots(len(classes), 3, figsize=(12, 2 * len(classes)), sharex=True)
    for k, c in enumerate(classes):
        for r, i in enumerate(np.flatnonzero(y == c)[:3]):
            ax[k, r].plot(R["x"][i, :, 0], R["x"][i, :, 1], ".", c="0.6", ms=2)
            ax[k, r].plot(R["x"][i, :, 0], R["xhat"][i], c="C0", lw=1.2)
            ax[k, r].invert_yaxis(); ax[k, r].set_ylabel(c, fontsize=8)
    fig.suptitle(f"{j}: observed (grey) and reconstructed from the latent mean (blue)")
    fig.tight_layout(); fig.savefig(FIGS / f"recon_{j}.png", dpi=100); plt.show()

# ---- latent space (Fig. 6) and how close it is to the N(0, I) prior ----
if (RUNS / "tcn" / "latent.npz").exists():
    Lz = np.load(RUNS / "tcn" / "latent.npz")
    rows_t = np.load(RUNS / "tcn" / "split.npz")["rows"]
    yz = meta.Type.values[rows_t]
    sub = np.random.default_rng(0).choice(len(yz), min(6000, len(yz)), replace=False)
    d = Lz["mu"].shape[1]
    fig, ax = plt.subplots(d, d, figsize=(11, 11))
    for a in range(d):
        for b in range(d):
            if a == b:
                for c in classes:
                    ax[a, b].hist(Lz["mu"][sub][yz[sub] == c, a], bins=40, histtype="step", label=c)
            elif a > b:
                for c in classes:
                    m = yz[sub] == c
                    ax[a, b].scatter(Lz["mu"][sub][m, b], Lz["mu"][sub][m, a], s=2, alpha=.5)
            else:
                ax[a, b].axis("off")
    ax[0, 0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.1, 1))
    fig.suptitle("tcn: latent means mu (their Fig. 6)"); fig.tight_layout()
    fig.savefig(FIGS / "fig6_latent.png", dpi=100); plt.show()
    stats = pd.DataFrame({"mean of mu": Lz["mu"].mean(0), "std of mu": Lz["mu"].std(0),
                          "mean posterior sigma": Lz["std"].mean(0)}, index=[f"z{i}" for i in range(d)]).round(3)
    stats.to_csv(TABLES / "latent_stats.csv")
    print("prior N(0, I) would give: mean 0, std 1; posterior sigma near 1 means the KL term is active\n",
          stats.to_string())

# ---- every model: its latent codes against the N(0, I) prior ----
from scipy.stats import chi2
rows = []
for j in jobs:
    if not (RUNS / j / "latent.npz").exists():
        continue
    Lj = np.load(RUNS / j / "latent.npz")
    mu = Lj["mu"].astype(float)
    m, cov, corr = mu.mean(0), np.cov(mu, rowvar=False), np.corrcoef(mu, rowvar=False)
    z = np.random.default_rng(0).standard_normal((100_000, mu.shape[1]))          # samples from the prior
    d2 = np.einsum("ij,jk,ik->i", z - m, np.linalg.inv(cov), z - m)              # distance to the encoded data
    rows.append({"job": j, "largest |mean of mu|": np.abs(m).max(), "std of mu (min)": mu.std(0).min(),
                 "std of mu (max)": mu.std(0).max(), "mean posterior sigma": Lj["std"].mean(),
                 "largest |corr| between dims": np.abs(corr - np.eye(len(m))).max(),
                 "prior samples outside the data's 99% region (%)": 100 * np.mean(d2 > chi2.ppf(0.99, len(m)))})
if rows:
    prior = pd.DataFrame(rows).round(3)
    prior.to_csv(TABLES / "latent_vs_prior.csv", index=False)
    print("\nlatent codes against the prior, per model (the region is a Gaussian fitted to the latent means):\n",
          prior.to_string(index=False))
