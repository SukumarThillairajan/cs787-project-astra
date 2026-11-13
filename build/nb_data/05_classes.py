# ---- 2 · Classes: imbalance, example curves, and what pre-processing does ----
tables = {name: audit.class_table(Z, SEED) for name, Z in Z_ALL.items()}
for name, t in tables.items():
    show(f"{name}: curves per class, and the two splits (ours = over stars, split {SEED}; paper = stored in the file)",
         t, f"{name}_classes.csv")
safely("imbalance figure", lambda: audit.plot_imbalance(tables, FIGS / "class_imbalance.png"))
for name, Z in Z_ALL.items():
    safely(f"{name} gallery", lambda: audit.plot_gallery(Z, name, FIGS / f"{name}_gallery.png"))

if WITH_PIPELINE and "catalina" in Z_ALL:
    def pipeline():
        cat = download(catalina.CATALOG_URL, WORK / "data" / "CatalinaVars.tbl")
        phot = download(catalina.PHOT_URL, WORK / "data" / "AllVar.phot.gz")
        audit.pipeline_example(cat, phot, Z_ALL["catalina"], FIGS / "catalina_pipeline.png")
    safely("pipeline figure", pipeline)
