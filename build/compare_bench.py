"""The four-way comparison of the benchmark, offline (no GPU), from downloaded result folders.

    python build/compare_bench.py <folder> <dataset> [--data <catalina_L100.npz or ztf_L40.npz>] [--out <dir>]

<folder> holds the results_<dataset>/ folders of the benchmark notebooks (anywhere below it; one sub-folder per
notebook output is fine). Tables are printed and saved as CSV, figures as PNG, into --out
(default: <folder>/comparison_<dataset>).
With --data (the same data file the notebooks used) two more things are made: the side-by-side figure of real and
generated curves for the same conditions, and the fidelity scores of all families in ONE feature space (a feature
CNN per split is trained here on the CPU, a few minutes each; the notebooks' own fidelity columns each use that
notebook's feature CNN and are only roughly comparable).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra import bench_report as rep  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
args = sys.argv[1:]
opt = {k: args[args.index(k) + 1] for k in ("--data", "--out") if k in args}
root, dataset = Path(args[0]), args[1]
out = Path(opt.get("--out", root / f"comparison_{dataset}"))
out.mkdir(parents=True, exist_ok=True)
res = rep.collect([root], f"results_{dataset}")
assert res, f"no results_{dataset}/*.json found below {root}"
print(f"{len(res)} jobs: {sorted(res)}")
hashes = sorted({(r["seed"], r["data_hash"]) for r in res.values()})
if len(hashes) != len({s for s, _ in hashes}):
    print(f"!! not all results use the same data and split (split, hash): {hashes}; such results are not paired")
search = rep.collect([root], f"results_{dataset}", stage="search")
print(f"{len(search)} search runs: {sorted(search)}")


def show(title, table, name, index=False):
    if len(table):
        table.to_csv(out / name, index=index)
        print(f"\n{title}\n{table.to_string(index=index)}")


show("The search (split 0, validation only)", rep.search_table(search), "search.csv")
rep.plot_search(search, out / "search_curves.png")
show("Reference rows on real data", rep.real_table(res), "real_rows.csv")
show("All jobs: selected checkpoints, test scores %", rep.job_table(res), "jobs.csv")
show("All families: mean ± std over the splits", rep.family_table(res), "families.csv")
show("If training had stopped after 5 min ... 10 h on one T4 (test TSTR)", rep.budget_table(res), "budgets.csv")
show("Conditions no training star has (one star's phases, another's amplitude), test TSTR", rep.novel_table(res),
     "novel_conditions.csv")
show("Per-class test recall %", rep.class_table(res), "per_class_recall.csv", True)
show("Paired differences between families (balanced accuracy, % points)", rep.family_pairs(res), "pairs.csv")
show("Other sampling settings (3 runs each)", rep.sweep_table(res), "sampling_sweeps.csv")
models = {j: r for j, r in res.items() if r["family"] != "real"}
rep.plot_headline(res, out / "headline.png")
rep.plot_class_bars(res, out / "per_class.png")
rep.plot_budget(res, out / "budgets.png")
rep.plot_training({j: r for j, r in models.items() if r["seed"] == 0}, out / "training_split0.png")
rep.plot_sweeps(models, out / "sampling_sweeps.png")
if "--data" in opt:
    Z = np.load(opt["--data"])
    for seed in sorted({r["seed"] for r in models.values()}):
        rep.plot_family_samples(res, Z, out / f"samples_split{seed}.png", seed=seed)
    show("Fidelity of every job's saved samples in ONE feature space",
         rep.fidelity_one_space(res, opt["--data"], out, "cpu"), "fidelity_one_space.csv")
else:
    print("\n(no --data given: the side-by-side samples figure and the one-space fidelity table are skipped)")
print(f"\nsaved to {out}")
