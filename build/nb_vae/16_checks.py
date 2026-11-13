# ---- Deliverables check ----
need = [TABLES / "table1_class_counts.csv", FIGS / "data_examples.png", TABLES / "fig4_losses.csv"]
summary = {"session": SESSION, "smoke": SMOKE, "hours": round((time.time() - RUN_STARTED) / 3600, 2), "jobs": {}}
for j in jobs:
    st = load(RUNS / j / "status.json") or {"state": "not started"}
    summary["jobs"][j] = st
    if st["state"] == "done":
        need += [RUNS / j / f for f in ("model.pt", "history.json", "latent.npz", "recon.npz", "split.npz",
                                         "config.json", "log.txt")]
if all(summary["jobs"].get(j, {}).get("state") == "done" for j in ("tcn", "tcn_phys")):
    need += [TABLES / "table3_regression.csv", TABLES / "latent_vs_prior.csv", FIGS / "fig8_rrlyr_teff_tcn.png",
             FIGS / "fig6_latent.png", FIGS / "prior_samples_tcn.png"]
missing = [str(p.relative_to(WORK)) for p in need if not p.exists()]
json.dump(summary, open(TABLES / f"session{SESSION}_summary.json", "w"), indent=1, default=str)
print(json.dumps(summary["jobs"], indent=1, default=str))
print(f"\noutput size {sum(f.stat().st_size for f in WORK.rglob('*') if f.is_file()) / 1e9:.2f} GB")
print("ALL DELIVERABLES PRESENT" if not missing else f"MISSING: {missing}")
left = [j for j, s in summary["jobs"].items() if s["state"] != "done"]
print("unfinished:", left if left else "none")
