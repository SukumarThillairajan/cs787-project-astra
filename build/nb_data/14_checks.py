# ---- Deliverables check ----
need = [TABLES / "datasets_summary.csv", FIGS / "class_imbalance.png"]
for name in Z_ALL:
    need += [FIGS / f"{name}_{f}.png" for f in ("gallery", "period_amplitude", "sampling", "noise", "confusion")]
    need += [TABLES / f"{name}_{f}.csv" for f in ("classes", "leakage", "test_power", "confusion")]
if WITH_PIPELINE and "catalina" in Z_ALL:
    need.append(FIGS / "catalina_pipeline.png")
if WITH_OGLE:
    need += [FIGS / "ogle_stars_vs_curves.png", TABLES / "ogle_classes.csv"]
missing = [str(p.relative_to(WORK)) for p in need if not p.exists()]
print(f"datasets audited: {list(Z_ALL)} | {len(list(FIGS.glob('*.png')))} figures, {len(list(TABLES.glob('*.csv')))} tables "
      f"| {(time.time() - RUN_STARTED) / 60:.1f} min")
print("ALL DELIVERABLES PRESENT" if not missing else f"MISSING: {missing}")
