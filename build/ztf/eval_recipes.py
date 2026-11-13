"""Row A of Table 5 (classifier trained on the real, oversampled training set; paper: 72.61 +- 0.69 on ZTF) for
variants of the ZTF pre-processing: which choices move the baseline, and how much does a new random draw of
stars move it? CPU, 3 runs x 5 CNNs per variant and data seed.

    python build/ztf/eval_recipes.py data/ztf <variant[,variant...]> [data seeds, e.g. 0,1,2] [threads] [batch]

batch: the classifier's batch size (default 128, the notebook's; the thesis uses 256). Rows are then named <variant>_b<batch>.

Variants. "ALeRCE" = Multiband_period of lc_classifier_1.2.1-P; "label" = the label file's catalogue period:
    notebook        astra.ztf.preprocess as it is (seed 0 must equal ztf_L40.npz, asserted): one curve per
                    (star, band) in g and r / label period / detections up to MJD 59048.4, repeated epochs dropped /
                    Table 2 counts drawn at random
    pairs           the same rule built by this script (other random draws than notebook)
    pairs_all       as pairs, every qualifying curve (DSCT 768, CEP 716; test 143 per class)
  Earlier recipes, all g only, one curve per star, Table 2 counts unless noted:
    g_eb_x2         ALeRCE, doubled for the eclipsing binaries / up to 2021-12-31 (the file of 2026-10-04 noon,
                    ztf_L40_g_2021_ebx2.npz)
    alerce_period   as g_eb_x2 without the doubling (same stars: a matched pair)
    cat             as g_eb_x2 with the catalogue periods of <dir>/catalogue_periods.csv (column p_cat)
    all_dets        ALeRCE / all detections up to 2026 (session 3, ztf_L40_all_detections.npz)
    to_2020-12, to_2021-06      ALeRCE / detections up to those dates
    tharun_period   as all_dets, with the period version Tharun's text sort picks (same stars as all_dets)
    best_band, best_band_2021   as all_dets / alerce_period, with, per star, the band that has most points
    merged_2020     g and r merged into one curve, ALeRCE, detections up to 2020-12-31
    tharun          band with most points / the version his sort picks / every star / all (his notebook's recipe)
    p_cut, p_all    a matched pair: the same stars and split, 40 points drawn from detections up to 2021-12-31
                    (p_cut) or from all detections (p_all), ALeRCE
Appends one line per variant and seed to <dir>/recipe_eval.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astra import ztf  # noqa: E402
from astra.classifier import fit_ensembles  # noqa: E402

Y2021 = 59579.0                                                   # 2021-12-31
VARIANTS = {"g_eb_x2": ("g", "paper_eb_x2", True, Y2021), "alerce_period": ("g", "paper", True, Y2021),
            "cat": ("g", "cat", True, Y2021), "all_dets": ("g", "paper", True, None),
            "to_2020-12": ("g", "paper", True, 59214.0), "to_2021-06": ("g", "paper", True, 59395.0),
            "tharun_period": ("g", "trans", True, None), "best_band": ("best", "paper", True, None),
            "best_band_2021": ("best", "paper", True, Y2021), "merged_2020": ("g+r", "paper", True, 59214.0),
            "tharun": ("best", "trans", False, None),
            "pairs": ("pairs", "label", True, ztf.MJD_MAX), "pairs_all": ("pairs", "label", False, ztf.MJD_MAX),
            "p_cut": ("g", "paper", True, Y2021, Y2021), "p_all": ("g", "paper", True, None, Y2021)}


def build(z, use, period, L, seed, match, sel=None):
    """The one-curve-per-star pre-processing of the earlier recipes, with the detection mask and period passed in.
    sel: the mask that decides which stars qualify, when it differs from `use`. The stars and the split then come
    from their own random streams, so two calls that differ only in `use` hold the same stars."""
    ok = use & (z["mag"] < ztf.MAG_THR)
    obj, mjd, mag = z["obj"][ok], z["mjd"][ok], z["mag"][ok]
    cnt = np.bincount(obj, minlength=len(z["oid"]))
    start = np.concatenate([[0], np.cumsum(cnt)])
    good = (cnt >= L) & np.isfinite(period) & (period > 0)
    rng = rng_pts = rng_split = np.random.default_rng(seed)
    if sel is not None:
        good &= np.bincount(z["obj"][sel & (z["mag"] < ztf.MAG_THR)], minlength=len(z["oid"])) >= L
        rng_pts, rng_split = np.random.default_rng(seed + 10_000), np.random.default_rng(seed + 20_000)
    chosen = []
    for c in ztf.CLASSES:
        idx = np.flatnonzero(good & (z["cls"] == ztf.ALERCE_NAME[c]))
        chosen.append(np.sort(rng.choice(idx, min(ztf.PAPER_COUNTS[c], len(idx)) if match else len(idx), replace=False)))
    y = np.concatenate([np.full(len(v), i) for i, v in enumerate(chosen)])
    chosen = np.concatenate(chosen)
    ph, x = np.zeros((len(chosen), L), np.float32), np.zeros((len(chosen), L), np.float32)
    for i, o in enumerate(chosen):
        s = rng_pts.choice(start[o + 1] - start[o], L, replace=False) + start[o]
        p = np.mod(mjd[s], period[o]) / period[o]
        order = np.argsort(p, kind="stable")
        ph[i], x[i] = p[order], mag[s][order] - np.median(mag[s])
    n_hold = int(0.2 * np.bincount(y).min())
    split = np.full(len(y), ztf.TRAIN, np.int8)
    for c in range(len(ztf.CLASSES)):
        idx = rng_split.permutation(np.flatnonzero(y == c))
        split[idx[:n_hold]], split[idx[n_hold:2 * n_hold]] = ztf.VAL, ztf.TEST
    return ph, x, y, split


def main():
    d = Path(sys.argv[1])
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0]
    torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 4)
    batch = int(sys.argv[5]) if len(sys.argv) > 5 else 128
    z = np.load(d / "ztf_periodic_raw.npz")
    z = {k: z[k] for k in ("oid", "cls", "obj", "mjd", "fid", "mag", "period")}
    n = len(z["oid"])

    def band_mask(band, cut):
        t = np.ones(len(z["mjd"]), bool) if cut is None else z["mjd"] <= cut
        if band != "best":
            return t & (z["fid"] == 1 if band == "g" else z["fid"] <= 2)
        cnts = np.stack([np.bincount(z["obj"][t & (z["fid"] == f) & (z["mag"] < ztf.MAG_THR)], minlength=n)
                         for f in (1, 2, 3)], 1)
        bfid = np.where(cnts[:, 1] == cnts.max(1), 2, cnts.argmax(1) + 1)      # r on ties, as in Tharun's notebook
        return t & (z["fid"] == bfid[z["obj"]])

    def pairs_view(cut):
        """One pseudo-star per (star, band): valid g and r detections up to `cut`, repeated epochs dropped."""
        lab = pd.read_csv(d / "ZTF_labels_v8.0.1.csv", low_memory=False).drop_duplicates("oid").set_index("oid")
        lab = lab.reindex(z["oid"])
        c = lab.classALeRCE.map({"EB/EW": "E", "EA": "E", "RRL": "RRL", "LPV": "LPV", "DSCT": "DSCT", "Ceph": "CEP"})
        t = (z["fid"] <= 2) & (z["mjd"] <= cut) & (z["mag"] < ztf.MAG_THR)
        obj2, mjd, mag = z["obj"][t].astype(np.int64) * 2 + (z["fid"][t] - 1), z["mjd"][t], z["mag"][t]
        order = np.lexsort((mjd, obj2))
        obj2, mjd, mag = obj2[order], mjd[order], mag[order]
        first = np.r_[True, (obj2[1:] != obj2[:-1]) | (mjd[1:] != mjd[:-1])]
        z2 = {"oid": np.repeat(z["oid"], 2), "cls": np.repeat(c.fillna("").values, 2), "obj": obj2[first],
              "mjd": mjd[first], "mag": mag[first]}
        return z2, np.repeat(pd.to_numeric(lab.period, errors="coerce").values, 2)

    per = pd.read_csv(d / "ztf_periods_all_versions.csv.gz")
    last = per.sort_values("version", kind="stable").drop_duplicates("oid", keep="last").set_index("oid")["value"]
    period = {"paper": z["period"], "trans": last.reindex(z["oid"]).values,
              "paper_eb_x2": z["period"] * np.where(z["cls"] == "E", 2.0, 1.0)}
    if (d / "catalogue_periods.csv").exists():
        period["cat"] = pd.read_csv(d / "catalogue_periods.csv").set_index("oid")["p_cat"].reindex(z["oid"]).values
    for name in sys.argv[2].split(","):
        if name != "notebook":
            band, pk, match, cut, *sel_cut = VARIANTS[name]
            if band == "pairs":
                zz, pp = pairs_view(cut)
                mask, sel = np.ones(len(zz["mjd"]), bool), None
            else:
                zz, pp, mask = z, period[pk], band_mask(band, cut)
                sel = band_mask(band, sel_cut[0]) if sel_cut else None
        for seed in seeds:
            if name == "notebook":
                out, _ = ztf.preprocess(d / "ztf_periodic_raw.npz", seed=seed, log=lambda *a: None)
                ph, x, y, split = out["phase"], out["mag"], out["y"], out["split"]
                if seed == 0:
                    ref = np.load(d / "ztf_L40.npz")
                    assert np.array_equal(ref["mag"], x) and np.array_equal(ref["split"], split), "ztf_L40.npz is stale"
            else:
                ph, x, y, split = build(zz, mask, pp, 40, seed, match, sel)
            parts = [{"x": torch.from_numpy(x[split == s]), "ph": torch.from_numpy(ph[split == s]),
                      "y": torch.from_numpy(y[split == s]).long()} for s in (ztf.TRAIN, ztf.VAL, ztf.TEST)]
            t0 = time.time()
            r = fit_ensembles(*parts, len(ztf.CLASSES), runs=3, ens=5, seed=0, device="cpu", amp=False,
                              batch=batch, log=lambda *a: None)
            row = {"variant": name if batch == 128 else f"{name}_b{batch}", "data_seed": seed, "n": len(y), "test_per_class": int(np.sum(split == ztf.TEST)) // 5,
                   "acc": round(100 * float(np.mean(r["test_acc"])), 2), "std_runs": round(100 * float(np.std(r["test_acc"])), 2),
                   **{f"recall_{c}": round(100 * float(np.mean(r["test_recall_per_class"], 0)[i]), 1)
                      for i, c in enumerate(ztf.CLASSES)}, "minutes": round((time.time() - t0) / 60, 1)}
            print(row, flush=True)
            out = d / "recipe_eval.csv"
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
