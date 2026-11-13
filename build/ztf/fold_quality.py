"""How clean is the folded curve? Our periods were estimated by ALeRCE around the paper's time, but the raw file
holds detections up to 2026, and a small period error grows with the time span. This measures the von Neumann
ratio (mean squared difference of phase-neighbours / variance; about 2 for pure noise, lower = cleaner) of 40
random g points per star, for several cut-off dates and for three period versions.

    python build/ztf/fold_quality.py data/ztf
"""
import sys

import numpy as np
import pandas as pd

d = sys.argv[1]
z = np.load(f"{d}/ztf_periodic_raw.npz")
cls, n = z["cls"], len(z["cls"])
m = (z["fid"] == 1) & (z["mag"] < 30)
o, t, x = z["obj"][m], z["mjd"][m], z["mag"][m]
cnt = np.bincount(o, minlength=n)
start = np.concatenate([[0], np.cumsum(cnt)])
per = pd.read_csv(f"{d}/ztf_periods_all_versions.csv.gz")
last = per.sort_values("version", kind="stable").drop_duplicates("oid", keep="last").set_index("oid")["value"]
PERIODS = {"1.2.1-P": z["period"], "newest": z["period_new"], "text-sorted last": last.reindex(z["oid"]).values}
CUTS = [("2020-06", 59009.5), ("2021-06", 59395), ("2021-12", 59579), ("2022-12", 59944), ("2023-12", 60309),
        ("2024-12", 60675), ("all", 1e9)]


def eta(tt, xx, p):
    v = xx[np.argsort(np.mod(tt, p) / p)]
    return np.mean(np.diff(v) ** 2) / v.var() if v.var() > 0 and np.isfinite(p) else np.nan


rng, rows = np.random.default_rng(0), []
for c in ("E", "RRL", "LPV", "DSCT", "CEP"):
    cand = np.flatnonzero((cls == c) & (cnt >= 80))
    res = {f"1.2.1-P <= {k}": [] for k, _ in CUTS} | {f"{k} all": [] for k in ("newest", "text-sorted last")}
    for i in rng.choice(cand, min(2000, len(cand)), replace=False):
        tt, xx = t[start[i]:start[i + 1]], x[start[i]:start[i + 1]]
        if (tt <= CUTS[0][1]).sum() < 40:                               # the same stars in every column
            continue
        for k, cut in CUTS:
            j = rng.choice(np.flatnonzero(tt <= cut), 40, replace=False)
            res[f"1.2.1-P <= {k}"].append(eta(tt[j], xx[j], PERIODS["1.2.1-P"][i]))
            if k == "all":
                for name in ("newest", "text-sorted last"):
                    res[f"{name} all"].append(eta(tt[j], xx[j], PERIODS[name][i]))
    rows.append({"class": c, "stars": len(res["1.2.1-P <= all"]), **{k: round(float(np.nanmedian(v)), 3) for k, v in res.items()}})
pd.set_option("display.width", 250)
print("median von Neumann ratio; columns: period version, detections used")
print(pd.DataFrame(rows).to_string(index=False))
