"""The ZTF dataset of Garcia-Jara et al. (2022), rebuilt from public sources.

What the paper states (Sec. 3): the periodic variable stars of ALeRCE's ZTF training set (Sanchez-Saez et al.
2021; classes EB, RRL, LPV, DSCT, CEP), light curves "processed by the ALeRCE broker", anomalous observations
(marked with magnitude 100) removed with a threshold of 30, folded with the provided period, 40 randomly
chosen observations per curve, median-centred. Table 2 gives the class counts after pre-processing.

What it does not state: the photometric band, the date of the data, the period's origin, how stars were
selected. We recovered these from the paper itself (notes/ztf_exact_dataset_leads.md): the real light curves
plotted in the paper and in the first author's thesis were digitised and matched to ALeRCE's detections, which
identified the 21 stars shown. They tell us that
  - a light curve is the detections of one star in ONE band, and the dataset holds g curves and r curves
    (14 and 7 of the 21);
  - the period is the `period` column of ALeRCE's internal label file, i.e. the period of the catalogue the
    label came from (equal to 1e-5 for all 21), with phase = (MJD mod T) / T. Most Gaia-labelled stars have no
    period there (78% of the RR Lyrae, 74% of the LPVs), cannot be folded, and drop out;
  - no plotted point is later than MJD 59048.3 (2020-07-19), and many are that late: one snapshot of that date;
  - only the magnitude threshold is applied; the 40 points are centred on their own median.
With one curve per (star, band), stars that have a period, detections up to that date and repeated epochs
dropped, "at least 40 points" gives EB 31,174 / RRL 18,748 / LPV 5,470, against Table 2's 31,477 / 18,729 /
5,245. DSCT and CEP come out at 768 and 716 against 507 and 471, which we cannot explain. So:
  - light curves = ALeRCE detections (TAP table ztf.detection), corrected magnitudes magpsf_corr < 30, per band;
  - keep curves with at least L points, then draw at random exactly the paper's number of curves per class
    (all of them where fewer qualify: EB), so the class imbalance and the split sizes match Table 2;
  - split as in Sec. 4.1: validation and test each hold 20% of the smallest class, for every class. As in the
    paper the split is over curves, so the g and the r curve of a star can fall on different sides.
"""
import numpy as np
import pandas as pd

CLASSES = ["EB", "RRL", "LPV", "DSCT", "CEP"]                    # the paper's names, in Table 2 order
ALERCE_NAME = {"EB": "E", "RRL": "RRL", "LPV": "LPV", "DSCT": "DSCT", "CEP": "CEP"}
PAPER_COUNTS = {"EB": 31477, "RRL": 18729, "LPV": 5245, "DSCT": 507, "CEP": 471}
TRAIN, VAL, TEST = 0, 1, 2
MAG_THR = 30.0
MJD_MAX = 59048.4                                                 # 2020-07-19
BANDS = (1, 2)                                                    # g, r


def preprocess(raw_npz, L=40, bands=BANDS, seed=0, match_paper_counts=True, period_key="period_label",
               mjd_max=MJD_MAX, log=print):
    """raw_npz: the file written by build/ztf/download_ztf.py and add_label_periods.py. bands: 1 = g, 2 = r.
    period_key: "period_label" (the label file's catalogue period) or "period" (ALeRCE's Multiband_period,
    which is half the orbital period for most eclipsing binaries).
    mjd_max: the last date of the detections used (None = all of them)."""
    z = np.load(raw_npz)
    period, cls = z[period_key], z["cls"]
    ok = np.isin(z["fid"], bands) & (z["mag"] < MAG_THR)
    if mjd_max is not None:
        ok &= z["mjd"] <= mjd_max
    # one light curve per (star, band): sort its detections by time and drop repeated epochs
    cur, mjd, mag, err = z["obj"][ok].astype(np.int64) * 4 + z["fid"][ok], z["mjd"][ok], z["mag"][ok], z["err_ext"][ok]
    order = np.lexsort((mjd, cur))
    cur, mjd, mag, err = cur[order], mjd[order], mag[order], err[order]
    new = np.r_[True, (cur[1:] != cur[:-1]) | (mjd[1:] != mjd[:-1])]
    cur, mjd, mag, err = cur[new], mjd[new], mag[new], err[new]
    ids, start, cnt = np.unique(cur, return_index=True, return_counts=True)
    star, band = ids // 4, ids % 4
    good = (cnt >= L) & np.isfinite(period[star]) & (period[star] > 0)
    rng = np.random.default_rng(seed)
    chosen, qualify = [], {}
    for c in CLASSES:
        idx = np.flatnonzero(good & (cls[star] == ALERCE_NAME[c]))
        qualify[c] = len(idx)
        k = min(PAPER_COUNTS[c], len(idx)) if match_paper_counts else len(idx)
        chosen.append(np.sort(rng.choice(idx, k, replace=False)))
    y = np.concatenate([np.full(len(v), i) for i, v in enumerate(chosen)])
    chosen = np.concatenate(chosen)
    n = len(chosen)
    out = dict(oid=z["oid"][star[chosen]], band=band[chosen].astype(np.int8), y=y, period=period[star[chosen]],
               phase=np.zeros((n, L), np.float32), mag=np.zeros((n, L), np.float32),
               err=np.zeros((n, L), np.float32), amp=np.zeros(n, np.float32))
    for i, k in enumerate(chosen):
        s = rng.choice(cnt[k], L, replace=False) + start[k]       # L random observations
        ph = np.mod(mjd[s], out["period"][i]) / out["period"][i]  # Eqs. 8-9
        o = np.argsort(ph, kind="stable")
        x = mag[s][o] - np.median(mag[s])                         # median centring
        out["phase"][i], out["mag"][i], out["err"][i] = ph[o], x, err[s][o]
        out["amp"][i] = (x.max() - x.min()) / 2
    counts = np.bincount(y, minlength=len(CLASSES))
    n_hold = int(0.2 * counts.min())
    split = np.full(n, TRAIN, np.int8)
    for c in range(len(CLASSES)):
        idx = rng.permutation(np.flatnonzero(y == c))
        split[idx[:n_hold]] = VAL
        split[idx[n_hold:2 * n_hold]] = TEST
    out["split"] = split
    out["classes"] = np.array(CLASSES)
    has_p = np.isfinite(period) & (period > 0)
    table = pd.DataFrame({"class": CLASSES, "labeled stars": [int(np.sum(cls == ALERCE_NAME[c])) for c in CLASSES],
                          "with a period": [int(np.sum((cls == ALERCE_NAME[c]) & has_p)) for c in CLASSES],
                          f"curves >= {L} points": [qualify[c] for c in CLASSES], "ours": counts,
                          "paper": [PAPER_COUNTS[c] for c in CLASSES]})
    for b, name in ((1, "g"), (2, "r")):
        table[name] = [int(((y == c) & (out["band"] == b)).sum()) for c in range(len(CLASSES))]
    table["train"] = [int(((y == c) & (split == TRAIN)).sum()) for c in range(len(CLASSES))]
    table["val"] = n_hold
    table["test"] = n_hold
    log(f"{n:,} light curves of {L} points ({int((out['band'] == 1).sum()):,} in g, {int((out['band'] == 2).sum()):,} in r)")
    return out, table
