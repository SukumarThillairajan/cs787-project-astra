# ---- decode latent vectors from three sources: an encoded real curve, the prior, the fitted class Gaussian ----
SHOW = ["CEP", "ECL", "LPV", "RRLYR"]
for j in ("tcn", "tcn_s14"):
    if not (RUNS / j / "model.pt").exists():
        continue
    vae = tcn_vae.VAE_TCN(4, 600, 5, 48, 9, 3, 0.2, 8, 0).to(DEV)
    vae.load_state_dict(torch.load(RUNS / j / "model.pt", map_location=DEV)); vae.eval()
    mu = np.load(RUNS / j / "latent.npz")["mu"]
    sp = np.load(RUNS / j / "split.npz")
    y_all = meta.Type.values[sp["rows"]]
    rng = np.random.default_rng(0)
    dt = torch.linspace(0, 1, 600, device=DEV).repeat(4, 1)
    fig, ax = plt.subplots(3, len(SHOW), figsize=(4 * len(SHOW), 7), sharex=True, sharey=True)
    for k, c in enumerate(SHOW):
        lab = torch.eye(8, device=DEV)[[classes.index(c)] * 4]
        tr_c, va_c = sp["train"][y_all[sp["train"]] == c], sp["val"][y_all[sp["val"]] == c]
        z_rec = mu[rng.choice(va_c, 4, replace=False)]
        z_prior = rng.standard_normal((4, 4))
        z_fit = rng.multivariate_normal(mu[tr_c].mean(0), np.cov(mu[tr_c], rowvar=False), 4)
        for r, z in enumerate((z_rec, z_prior, z_fit)):
            with torch.no_grad():
                lc = vae.decoder(torch.tensor(z, dtype=torch.float32, device=DEV), dt, lab)[:, :, 0].cpu().numpy()
            for i in range(4):
                ax[r, k].plot(dt[0].cpu().numpy(), lc[i], lw=1)
        ax[0, k].set_title(c)
    for r, name in enumerate(("z = encoded real curve", "z ~ N(0, I), the prior", "z ~ Gaussian fitted to the class")):
        ax[r, 0].set_ylabel(name, fontsize=9)
    ax[0, 0].invert_yaxis()
    fig.suptitle(f"{j}: curves decoded from three kinds of latent vectors (4 draws each)"); fig.tight_layout()
    fig.savefig(FIGS / f"prior_samples_{j}.png", dpi=100); plt.show()
