# ---- Deliverables check: is everything saved that the comparison and the report need? ----
load = lambda p: json.loads(p.read_text()) if p.exists() else None
need = [NPZ, TABLES / "split_counts.csv", FIGS / "data_examples.png", FIGS / "feature_cnn_curves.png"]
need += [RUNS / f"feature_cnn_s{s}.pt" for s in SEEDS]
lacking = []                                    # results that exist but miss a part
summary = {"session": SESSION, "smoke": SMOKE, "stage": STAGE, "family": FAMILY, "dataset": DATASET,
           "slots_per_gpu": SPG, "speeds": SPEEDS, "hours": round((time.time() - RUN_STARTED) / 3600, 2), "jobs": {}}
for j, runs in SHOWN:
    st = load(runs / j / "status.json") or {"state": "not started"}
    summary["jobs"][f"{runs.name}/{j}"] = st
    if st["state"] != "done":
        continue
    need += [RESULTS / f"{j}.json", runs / j / "config.json", runs / j / "log.txt"]
    if j.startswith("real_s"):
        need.append(RESULTS / f"{j}_pred.npz")
    elif st.get("stage") == "search":
        need += [runs / j / f for f in ("search.json", "metrics.json", "ckpts.json", "last.pt")]
    else:
        need += [runs / j / f for f in ("final.json", "metrics.json", "samples.npz", "train_curve.csv")]
        need += [RESULTS / f"{j}_pred.npz", RESULTS / f"{j}_samples.npz"]
        if "novel_selected" not in (load(runs / j / "final.json") or {}).get("parts", {}):
            lacking.append(f"{runs.name}/{j}: the novel-conditions scores")
states = [(j.split("/")[-1], s.get("stage"), s["state"]) for j, s in summary["jobs"].items()]
if any(st == "done" and stage == "search" for j, stage, st in states):
    need.append(TABLES / "search.csv")
if any(st == "done" and j.startswith("real_s") for j, _, st in states):
    need.append(TABLES / "real_rows.csv")
if any(st == "done" and stage == "final" and not j.startswith("real_s") for j, stage, st in states):
    need += [TABLES / "jobs.csv", TABLES / "budgets.csv", FIGS / "training_curves.png"]
missing = [str(p.relative_to(WORK)) for p in need if not p.exists()] + lacking
json.dump(summary, open(TABLES / f"session{SESSION}_{STAGE}_summary.json", "w"), indent=1, default=str)
print(json.dumps(summary["jobs"], indent=1, default=str))
out_gb = sum(f.stat().st_size for f in WORK.rglob("*") if f.is_file()) / 1e9
print(f"\noutput size {out_gb:.2f} GB (Kaggle keeps up to 20 GB)")
print("ALL DELIVERABLES PRESENT" if not missing else f"MISSING: {missing}")
left = [j for j, stage, st in states if st != "done" or (not SMOKE and not j.startswith("real_s") and stage not in (STAGE, "final"))]
print(f"next session: attach this version's output, set SESSION = {SESSION + 1} | {STAGE} stage unfinished:",
      left if left else "none")

# ---- one small file to send back after every session: results, tables, figures, logs (no model weights) ----
import zipfile
BUNDLE = WORK / f"report_{FAMILY}_{'smoke' if SMOKE else STAGE}_session{SESSION}.zip"
try:
    files = [f for root in (RESULTS, TABLES, FIGS) for f in root.rglob("*") if f.is_file()]
    files += [f for runs in STAGE_RUNS.values() for pat in ("*/*.json", "*/log.txt", "*/train_curve.csv", "*.json", "*.log")
              for f in runs.glob(pat)]
    with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(set(files)):
            z.write(f, f.relative_to(WORK))
    print(f"\nSEND BACK: {BUNDLE.name} ({BUNDLE.stat().st_size / 1e6:.1f} MB, in the Output tab) and this session's log")
except Exception as e:
    print(f"\n(the report bundle could not be written: {type(e).__name__}: {e}; send the folders instead)")
