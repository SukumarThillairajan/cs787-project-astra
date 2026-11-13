"""Which selection reproduces the paper's ZTF class counts (Table 2: EB 31477, RRL 18729, LPV 5245, DSCT 507,
CEP 471)? Explores the band, the date cut-off, and the class-range filter of Sec. 3.2.2 (drop a curve when more
than 90% of its magnitudes lie outside class median +- k * class std), followed by "at least 40 points".

    python build/ztf/calibrate_counts.py data/ztf
"""
import sys

import numpy as np
import pandas as pd

d = sys.argv[1]
z = np.load(f"{d}/ztf_periodic_raw.npz")
obj, mjd, fid, mag = z["obj"], z["mjd"], z["fid"], z["mag"]
cls = z["cls"]
n = len(cls)
PAPER = {"E": 31477, "RRL": 18729, "LPV": 5245, "DSCT": 507, "CEP": 471}
cid = np.array([list(PAPER).index(c) for c in cls])
ok = mag < 30

print("per class: median detections (mag < 30) in g / r up to 2021-06-30, and median magnitude in g / r")
m = ok & (mjd <= 59395)
for c in PAPER:
    sel = cls == c
    ng, nr = np.bincount(obj[m & (fid == 1)], minlength=n)[sel], np.bincount(obj[m & (fid == 2)], minlength=n)[sel]
    mg, mr = mag[m & (fid == 1) & sel[obj]], mag[m & (fid == 2) & sel[obj]]
    print(f"  {c:5s} n_g {np.median(ng):5.0f} n_r {np.median(nr):5.0f} | mag g {np.median(mg):.2f} +- {mg.std():.2f}, "
          f"r {np.median(mr):.2f} +- {mr.std():.2f}")

rows = []
for cut_name, cut in [("2020-06-09", 59009.5), ("2020-12-31", 59214), ("2021-06-30", 59395), ("2021-12-31", 59579), ("all", 1e9)]:
    for band, bsel in [("g", fid == 1), ("r", fid == 2), ("g+r", fid <= 2)]:
        m = ok & (mjd <= cut) & bsel
        o, x, c_of = obj[m], mag[m], cid[obj[m]]
        cnt = np.bincount(o, minlength=n)
        med = np.array([np.median(x[c_of == k]) for k in range(5)])
        std = np.array([x[c_of == k].std() for k in range(5)])
        dev = np.abs(x - med[c_of]) / std[c_of]
        for k in (None, 2.0, 1.5, 1.25, 1.0, 0.75):
            keep = cnt >= 40
            if k is not None:
                frac_out = np.bincount(o, weights=dev > k, minlength=n) / np.maximum(cnt, 1)
                keep = keep & (frac_out <= 0.9)
            c = {name: int(np.sum(keep & (cid == i))) for i, name in enumerate(PAPER)}
            e = float(np.sqrt(np.mean([np.log(max(c[q], 1) / PAPER[q]) ** 2 for q in PAPER])))
            rows.append({"cutoff": cut_name, "band": band, "class-range k": k, **c, "rms_log_err": round(e, 3)})
out = pd.DataFrame(rows).sort_values("rms_log_err")
print("\npaper:", PAPER)
print(out.head(25).to_string(index=False))
out.to_csv(f"{d}/calibration_table_classrange.csv", index=False)
