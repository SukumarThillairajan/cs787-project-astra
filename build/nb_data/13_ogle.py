# ---- 6 · The PELS-VAE training file (OGLE-III): stars against curves ----
if WITH_OGLE:
    def ogle():
        from astra import pelsvae
        _, meta = pelsvae.load_raw(WORK / "data")
        t = audit.ogle_table(meta)
        show("PELS-VAE training file: curves, stars, parameters, and copies across their random split", t, "ogle_classes.csv")
        print(f"\nall classes: {len(meta):,} curves of {meta.OGLE_id.nunique():,} stars")
        audit.plot_ogle(t, FIGS / "ogle_stars_vs_curves.png")
    safely("OGLE section", ogle)
