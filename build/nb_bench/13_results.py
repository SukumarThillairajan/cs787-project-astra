# ---- Tables and figures from everything finished so far ----
import traceback
from astra import bench_report as rep

pd.set_option("display.width", 250)


def safely(what, fn):
    """A failing table or figure must not cost the session's output: report it and go on."""
    try:
        return fn()
    except Exception:
        print(f"\n!! {what} failed:\n{traceback.format_exc()}")


def show(title, make, name, index=False):
    table = safely(title, make)
    if table is not None and len(table):
        table.to_csv(TABLES / name, index=index)
        print(f"\n{title}\n{table.to_string(index=index)}")


mine = lambda d: {j: r for j, r in d.items() if Path(r["_from"]).resolve() == RESULTS.resolve()}
res = rep.collect([WORK, INPUT], RESULTS.name)                          # final stage: this notebook's + attached
search = rep.collect([WORK, INPUT], RESULTS.name, stage="search")       # search stage (validation only)
own, own_search = mine(res), mine(search)
models = {j: r for j, r in own.items() if r["family"] != "real"}
if not own and not own_search:
    print("(no finished jobs yet)")

# ---- the search: every configuration on split 0, validation stars only ----
show(f"{rep.FAMILIES[FAMILY]}: the search on split 0 (validation TSTR balanced accuracy %, best first)",
     lambda: rep.search_table(own_search), "search.csv")
safely("search curves", lambda: rep.plot_search(own_search, FIGS / "search_curves.png"))

# ---- the final stage: the winner on five splits, test scores ----
show(f"Reference rows on real data ({DATASET}), test balanced accuracy %", lambda: rep.real_table(own), "real_rows.csv")
show(f"{rep.FAMILIES[FAMILY]} on {DATASET}: selected checkpoints, test scores % (copies = memorisation)",
     lambda: rep.job_table(own), "jobs.csv")
show("If training had stopped after 5, 10, 15, 30, 60 min or 2, 5, 10 h on one T4 (best checkpoint within, test TSTR)",
     lambda: rep.budget_table(own), "budgets.csv")
show("Conditions no training star has (one star's phases, another's amplitude), test TSTR, 3 runs each",
     lambda: rep.novel_table(own), "novel_conditions.csv")
show("Per-class test recall %, mean over the splits", lambda: rep.class_table(own), "per_class_recall.csv", True)
show("Other sampling settings of the selected checkpoints (3 runs each)", lambda: rep.sweep_table(own), "sampling_sweeps.csv")

safely("training curves", lambda: rep.plot_training(models, FIGS / "training_curves.png"))
for j in sorted(models):
    safely(f"samples of {j}", lambda: rep.plot_samples(
        STAGE_RUNS["final"] / j, Z, f"{j}: real vs generated, same condition (selected checkpoint)",
        FIGS / f"samples_{j}.png"))
safely("classifier curves", lambda: rep.plot_classifiers(own, FIGS / "classifier_curves.png"))
safely("sampling sweeps", lambda: rep.plot_sweeps(models, FIGS / "sampling_sweeps.png"))
safely("budget figure", lambda: rep.plot_budget(own, FIGS / "budgets.png"))
safely("headline figure", lambda: rep.plot_headline(own, FIGS / "headline.png"))

# ---- the four-way comparison (needs the other benchmark notebooks' outputs under Add Input) ----
others = sorted({r["family"] for r in res.values()} - {FAMILY, "real"})
if others:
    print(f"\n==== comparison with attached results: {[rep.FAMILIES.get(f, f) for f in others]} ====")
    hashes = sorted({(r["seed"], r["data_hash"]) for r in res.values()})
    if len(hashes) != len({s for s, _ in hashes}):
        print(f"!! the attached results do not all use the same data and split (split, hash): {hashes}\n"
              "   rows are still listed, but results with different hashes are not paired")
    feats = sorted({(r["seed"], r.get("feat_hash")) for r in res.values() if r.get("feat_hash")})
    if len(feats) != len({s for s, _ in feats}):
        print("note: the notebooks trained their own feature CNNs and these differ, so FID, density and recall of "
              "different notebooks are measured in slightly different feature spaces. "
              "The classifier scores (same test stars) are not affected.")
    show("All families: mean ± std over the splits", lambda: rep.family_table(res), "comparison_families.csv")
    show("All families: the compute-matched view", lambda: rep.budget_table(res), "comparison_budgets.csv")
    show("All families: conditions no training star has", lambda: rep.novel_table(res), "comparison_novel_conditions.csv")
    show("Per-class test recall %, all families", lambda: rep.class_table(res), "comparison_per_class.csv", True)
    show("Paired differences between families (balanced accuracy, % points)", lambda: rep.family_pairs(res),
         "comparison_pairs.csv")
    split0 = {j: r for j, r in res.items() if r["family"] != "real" and r["seed"] == 0}
    safely("comparison curves", lambda: rep.plot_training(split0, FIGS / "comparison_training_split0.png"))
    safely("comparison budgets", lambda: rep.plot_budget(res, FIGS / "comparison_budgets.png"))
    safely("comparison headline", lambda: rep.plot_headline(res, FIGS / "comparison_headline.png"))
    safely("comparison per class", lambda: rep.plot_class_bars(res, FIGS / "comparison_per_class.png"))
    safely("comparison samples", lambda: rep.plot_family_samples(res, Z, FIGS / "comparison_samples_split0.png"))
    show("Fidelity of every job's saved samples in ONE feature space (this notebook's feature CNNs)",
         lambda: rep.fidelity_one_space(res, NPZ, RUNS, DEV, log=lambda *a: None), "comparison_fidelity_one_space.csv")
elif STAGE == "final" or SMOKE:
    print("\n(four-way comparison: attach the other benchmark notebooks' outputs to see it)")
