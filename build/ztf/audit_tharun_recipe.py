"""What would Tharun's notebook (ZTF Dataset Generation Attempts/ztf-data-generation.ipynb) produce? Its logic,
replayed on the raw download we already have: band = the one with the most valid detections (r on ties), at
least 40 points, period = the Multiband_period row that sorts last by version string, no reduction to Table 2.
Also counts the repeated same-epoch rows of ALeRCE's detection table, which his finalize() trips over.

    python build/ztf/audit_tharun_recipe.py data/ztf [classifier_outputs.csv]

The optional second argument is ALeRCE_lc_classifier_outputs_ZTF_unlabeled_set_20200609.csv (Zenodo 4279623);
with it, the script also tests whether classifier agreement explains the paper's class counts.
"""
import sys

import numpy as np
import pandas as pd

d = sys.argv[1]
z = np.load(f"{d}/ztf_periodic_raw.npz")
obj, mjd, fid, mag, cls, oid = z["obj"], z["mjd"], z["fid"], z["mag"], z["cls"], z["oid"]
n = len(cls)
PAPER = {"E": 31477, "RRL": 18729, "LPV": 5245, "DSCT": 507, "CEP": 471}
ok = np.isfinite(mag) & (mag < 30)

# 1. band choice and class counts
cnts = np.stack([np.bincount(obj[ok & (fid == f)], minlength=n) for f in (1, 2, 3)], 1)
best = cnts.max(1)
bfid = np.where(cnts[:, 1] == best, 2, cnts.argmax(1) + 1)
keep = best >= 40
t = pd.DataFrame({"class": list(PAPER), "labeled": [int((cls == c).sum()) for c in PAPER],
                  "tharun": [int((keep & (cls == c)).sum()) for c in PAPER], "paper": list(PAPER.values())})
for f, name in ((1, "g"), (2, "r"), (3, "i")):
    t[name] = [int((keep & (cls == c) & (bfid == f)).sum()) for c in PAPER]
t["share %"] = (100 * t.tharun / t.tharun.sum()).round(1)
t["paper share %"] = (100 * t.paper / t.paper.sum()).round(1)
print(t.to_string(index=False))
print(f"total {t.tharun.sum():,} (paper {sum(PAPER.values()):,}); r band {100 * t.r.sum() / t.tharun.sum():.1f}%; "
      f"val = test = {int(0.2 * t.tharun.min())} per class under Sec. 4.1 (paper 94)")

# 2. period: sort_values("version").iloc[-1]
per = pd.read_csv(f"{d}/ztf_periods_all_versions.csv.gz")
last = per.sort_values("version", kind="stable").drop_duplicates("oid", keep="last").set_index("oid")
print("\nperiod version his sort picks:", last.version.value_counts().to_dict())
same = np.isclose(last["value"].reindex(oid).values, z["period"], rtol=1e-2)
print(f"equal (1%) to lc_classifier_1.2.1-P, the period astra/ztf.py uses: {same.sum():,} of {n:,}")

# 3. repeated epochs, and the stars his finalize() loses (drop_duplicates, then size != 40)
m = ok & (fid == bfid[obj])
o, t_, x = obj[m], mjd[m], mag[m]
for name, band in (("g", 1), ("r", 2)):
    b = ok & (fid == band)
    ob, tb, xb = obj[b], mjd[b], mag[b]
    rep = (ob[1:] == ob[:-1]) & (tb[1:] == tb[:-1])
    dm = np.abs(xb[1:] - xb[:-1])[rep]
    print(f"\nband {name}: {rep.sum():,} repeated-epoch rows of {b.sum():,} ({100 * rep.mean():.1f}%); "
          f"|dmag| median {np.median(dm):.4f}, identical {100 * np.mean(dm == 0):.0f}%, < 0.05 mag {100 * np.mean(dm < 0.05):.0f}%")
    yr = 2018 + ((tb[1:][rep] - 58119) // 365.25).astype(int)
    all_yr = 2018 + ((tb - 58119) // 365.25).astype(int)
    print("  share of rows that repeat the previous epoch, by year:",
          {int(y): round(100 * float(np.sum(yr == y)) / max(int(np.sum(all_yr == y)), 1), 1) for y in np.unique(all_yr)})
cnt = np.bincount(o, minlength=n)
start = np.concatenate([[0], np.cumsum(cnt)])
rng, lost = np.random.default_rng(0), 0
for i in np.flatnonzero(cnt >= 40):
    s = rng.choice(cnt[i], 40, replace=False) + start[i]
    lost += len(np.unique(np.stack([t_[s], x[s].astype(np.float64)], 1), axis=0)) < 40
print(f"\nfinalize(): about {lost:,} of {int((cnt >= 40).sum()):,} stars draw two identical rows and are dropped")

# 4. does the classifier output file explain Table 2?
if len(sys.argv) > 2:
    u = pd.read_csv(sys.argv[2], usecols=["oid", "predicted_class"]).set_index("oid").reindex(oid)
    agree = u.predicted_class.values == cls
    rows = []
    for cut_name, cut in (("2020-06-09", 59009.5), ("2020-12-31", 59214), ("2021-06-30", 59395), ("all", 1e9)):
        for band_name, bands in (("g", [1]), ("r", [2]), ("g+r", [1, 2])):
            k = (np.bincount(obj[ok & (mjd <= cut) & np.isin(fid, bands)], minlength=n) >= 40) & agree
            c = {q: int((k & (cls == q)).sum()) for q in PAPER}
            e = float(np.sqrt(np.mean([np.log(max(c[q], 1) / PAPER[q]) ** 2 for q in PAPER])))
            rows.append({"rule": f"{band_name}>=40 up to {cut_name}, predicted == label", **c, "rms_log_err": round(e, 3)})
    print(f"\nlabeled periodic stars in the file: {int(u.predicted_class.notna().sum()):,}; paper: {PAPER}")
    print(pd.DataFrame(rows).sort_values("rms_log_err").head(5).to_string(index=False))
