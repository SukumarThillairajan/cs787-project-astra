"""Every checkpoint of every searched configuration, from downloaded search outputs (validation side only).

    python build/search_ckpts.py <folder> [family ...]

<folder> holds the results_<dataset>/ folders of the benchmark notebooks (anywhere below it). Per configuration:
minutes on one T4 alone, validation TSTR balanced accuracy, rare-tier copy share and memorisation ratio, the model's
own training / validation loss, FID and recall. The test score that the search logs for the analysis is not shown:
nothing in the search may be read against the test stars.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra import bench_report as rep  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
root, fams = sys.argv[1], sys.argv[2:]
res = rep.collect([root], "results_catalina", stage="search") or rep.collect([root], "results_ztf", stage="search")
for job, r in sorted(res.items(), key=lambda kv: (kv[1]["family"], kv[0])):
    if fams and r["family"] not in fams:
        continue
    print(f"== {job}: stopped by {r.get('stop')}, trained {r.get('solo_h', np.nan):.2f} h, best ckpt {r.get('best_ckpt')}")
    print("   ckpt   min   val TSTR  ±runs  copies% mem   loss train / val    FID   recall")
    for a in r.get("metrics") or []:
        print(f"   {a['ckpt']:4d} {60 * a['solo_h']:6.1f} {100 * a['val_bal_acc']:8.2f} {100 * np.std(a['val_bal_acc_runs']):6.2f}"
              f" {100 * rep.rare_mean(r, a['copy_share']):7.1f} {rep.rare_mean(r, a['mem_ratio']):5.2f}"
              f"   {a['loss_train']:8.4f} / {a['loss_val']:8.4f} {a['fid']:6.2f} {np.nanmean(a['recall']):6.3f}"
              f"{'  <- best' if a['ckpt'] == r.get('best_ckpt') else ''}")
