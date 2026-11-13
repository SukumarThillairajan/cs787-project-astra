# ---- Deliverables check: is everything saved that later notebooks and the report need? ----
need = [NPZ, TABLES / "table2_class_counts.csv", RUNS / "feature_cnn.pt", FIGS / "data_examples.png",
        TABLES / "table5_ours_vs_paper.csv"]
jobs = [j for q in QUEUES.values() for j in q]
summary = {"session": SESSION, "smoke": SMOKE, "hours": round((time.time() - RUN_STARTED) / 3600, 2), "jobs": {}}
for j in jobs:
    st = load(RUNS / j / "status.json") or {"state": "not started"}
    summary["jobs"][j] = st
    if st["state"] == "done":
        need += [RUNS / "trtr" / "trtr.json"] if j == "trtr" else \
                [RUNS / j / f for f in ("final.json", "metrics.json", "config.json", "log.txt", "samples_g_score.npz")]
missing = [str(p.relative_to(WORK)) for p in need if not p.exists()]
json.dump(summary, open(TABLES / f"session{SESSION}_summary.json", "w"), indent=1, default=str)
print(json.dumps(summary["jobs"], indent=1, default=str))
out_gb = sum(f.stat().st_size for f in WORK.rglob("*") if f.is_file()) / 1e9
print(f"\noutput size {out_gb:.2f} GB (Kaggle keeps up to 20 GB)")
print("ALL DELIVERABLES PRESENT" if not missing else f"MISSING: {missing}")
left = [j for j, s in summary["jobs"].items() if s["state"] != "done"]
print("next session: attach this version's output, set SESSION =", SESSION + 1, "| unfinished:", left if left else "none")
