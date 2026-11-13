# ---- Table 3 (latent <- physical parameters) and Fig. 8 ----
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

PHYS2 = ["abs_Gmag", "teff_val", "Period"]
PAPER_T3 = {"tcn": {"Linear": 0.863, "RF": 0.299, "MLP": 0.863}, "tcn_phys": {"Linear": 0.794, "RF": 0.289, "MLP": 0.798}}
REG = {"Linear": lambda: LinearRegression(fit_intercept=False),
       "RF": lambda: RandomForestRegressor(n_estimators=100, max_features=3, n_jobs=-1, random_state=0),
       "MLP": lambda: MLPRegressor(hidden_layer_sizes=(50,), activation="tanh", solver="adam", alpha=1e-4,
                                   learning_rate_init=1e-3, max_iter=200 if not SMOKE else 20, random_state=0)}
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
tcn_jobs = [j for j in jobs if j.startswith("tcn") and (RUNS / j / "latent.npz").exists()]
t3, fitted = [], {}
for j in tcn_jobs:
    print(f"[{(time.time() - RUN_STARTED) / 3600:5.2f} h] Table 3: fitting the three regressors for {j}", flush=True)
    Lz = np.load(RUNS / j / "latent.npz")
    m = meta.iloc[np.load(RUNS / j / "split.npz")["rows"]].reset_index(drop=True)
    rng = np.random.default_rng(0)
    # (a) one row per star, split over stars: validation stars are never seen in training
    keep = m.dropna(subset=PHYS2).drop_duplicates("OGLE_id").index.values
    P, MU, SD = m.loc[keep, PHYS2].values.astype(float), Lz["mu"][keep].astype(float), Lz["std"][keep].astype(float)
    perm = rng.permutation(len(keep)); ntr = int(0.8 * len(keep))
    itr, iva = perm[:ntr], perm[ntr:]
    # (b) every curve, split over curves: the augmented copies of a star land on both sides (as in their VAE split)
    rows_c = m.dropna(subset=PHYS2).index.values
    Pc, MUc = m.loc[rows_c, PHYS2].values.astype(float), Lz["mu"][rows_c].astype(float)
    perm_c = rng.permutation(len(rows_c)); ntr_c = int(0.8 * len(rows_c))
    ctr, cva = perm_c[:ntr_c], perm_c[ntr_c:]
    ids = m.OGLE_id.values[rows_c]
    shared = np.isin(ids[cva], ids[ctr]).mean()
    # (c) their dummy variant: 10 latent samples per star, a U(0,1) dummy input
    rep = lambda i: (np.hstack([np.repeat(P[i], 10, 0), rng.uniform(size=(10 * len(i), 1))]),
                     rng.normal(np.repeat(MU[i], 10, 0), np.repeat(SD[i], 10, 0)))
    (Ptr_d, Ztr_d), (Pva_d, Zva_d) = rep(itr), rep(iva)
    for name, make in REG.items():
        r = make().fit(P[itr], MU[itr])
        rc = make().fit(Pc[ctr], MUc[ctr])
        rd = make().fit(Ptr_d, Ztr_d)
        fitted[(j, name)] = r
        t3.append({"model": j, "regressor": name, "paper (Table 3)": PAPER_T3["tcn_phys" if "phys" in j else "tcn"][name],
                   "train RMSE (in-sample)": rmse(r.predict(P[itr]), MU[itr]),
                   "val RMSE (new stars)": rmse(r.predict(P[iva]), MU[iva]),
                   "val RMSE (split over curves)": rmse(rc.predict(Pc[cva]), MUc[cva]),
                   "val RMSE, dummy x10": rmse(rd.predict(Pva_d), Zva_d),
                   "predict the mean": rmse(MU[iva], MU[itr].mean(0)), "stars": len(keep),
                   "val curves with a copy in training (%)": round(100 * shared, 1)})
t3 = pd.DataFrame(t3).round(3)
t3.to_csv(TABLES / "table3_regression.csv", index=False)
print("Table 3 — RMSE in latent units (lower is better; 'predict the mean' is the score of knowing nothing)\n",
      t3.to_string(index=False) if len(t3) else "(no latents yet)")

# ---- Fig. 8: RR Lyrae at increasing T_eff (M_G = -0.24, P = 0.45 d), decoded with the cVAE (both seeds) ----
for j in ("tcn", "tcn_s14"):
    if not ((RUNS / j / "model.pt").exists() and (j, "RF") in fitted):
        continue
    vae = tcn_vae.VAE_TCN(4, 600, 5, 48, 9, 3, 0.2, 8, 0).to(DEV)
    vae.load_state_dict(torch.load(RUNS / j / "model.pt", map_location=DEV)); vae.eval()
    teff = np.linspace(6000, 7500, 5)
    pp = np.stack([np.full(5, -0.24), teff, np.full(5, 0.45)], 1)
    dt = torch.linspace(0, 1, 600, device=DEV).repeat(5, 1)
    lab = torch.eye(8, device=DEV)[[classes.index("RRLYR")] * 5]
    fig, ax = plt.subplots(3, 5, figsize=(18, 7), sharex=True, sharey=True)
    for r, name in enumerate(REG):
        z = torch.tensor(fitted[(j, name)].predict(pp), dtype=torch.float32, device=DEV)
        with torch.no_grad():
            lc = vae.decoder(z, dt, lab)[:, :, 0].cpu().numpy()
        for k in range(5):
            ax[r, k].plot(dt[0].cpu().numpy(), lc[k], "k.", ms=2)
            ax[0, k].set_title(f"T_eff = {teff[k]:.0f} K")
        ax[r, 0].set_ylabel(name)
    ax[0, 0].invert_yaxis()
    fig.suptitle(f"{j}: generated RR Lyrae as T_eff rises (their Fig. 8); rows = regressor"); fig.tight_layout()
    fig.savefig(FIGS / f"fig8_rrlyr_teff_{j}.png", dpi=100); plt.show()
