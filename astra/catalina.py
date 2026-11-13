"""Catalina Surveys DR1 periodic variables, pre-processed exactly as in Garcia-Jara et al. (2022), Sec. 3.

Steps: period folding -> outlier filtering (sliding z-score, then a class-range filter) -> keep curves with
at least L points -> random sub-sampling to L points -> median centering. Then the paper's split: validation
and test each hold 20% of the smallest class, for every class (Sec. 4.1).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import download

CATALOG_URL = "http://nesssi.cacr.caltech.edu/DataRelease/CatalinaVars.tbl"
PHOT_URL = "http://nesssi.cacr.caltech.edu/DataRelease/AllVar.phot.gz"

# Table 1 of the paper: Drake et al. (2014) type id -> new class
CLASS_MAP = {1: "EBSD/D", 3: "EBSD/D",                      # EW, beta Lyrae
             4: "RRL", 5: "RRL", 6: "RRL", 7: "RRL",        # RRab, RRc, RRd, Blazhko
             2: "EBC",                                      # EA
             13: "LPV",
             11: "DSCT", 12: "DSCT",                        # HADS, LADS
             9: "CEP", 10: "CEP"}                           # ACEP, Cep-II
CLASSES = ["EBSD/D", "RRL", "EBC", "LPV", "DSCT", "CEP"]
PAPER_COUNTS = {"EBSD/D": 28980, "RRL": 7533, "EBC": 4500, "LPV": 483, "DSCT": 241, "CEP": 182}   # Table 2
TRAIN, VAL, TEST = 0, 1, 2


def read_catalog(path):
    cols = ["css_id", "numid", "ra", "dec", "vmag", "period", "amp_cat", "nobs", "vartype"]
    cat = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=cols)
    cat = cat[cat.vartype.isin(list(CLASS_MAP))].copy()
    cat["period"] = pd.to_numeric(cat.period)          # "\nodata" only occurs in type 17, dropped above
    cat["cls"] = cat.vartype.map(CLASS_MAP)
    return cat.sort_values("numid").reset_index(drop=True)


def read_photometry(path, numids):
    """All (MJD, mag, err) rows of the wanted objects, grouped per object."""
    ph = pd.read_csv(path, header=None, usecols=[0, 1, 2, 3], names=["numid", "mjd", "mag", "err"],
                     dtype={"numid": np.int64, "mjd": np.float64, "mag": np.float64, "err": np.float64})
    ph = ph[ph.numid.isin(set(numids))].sort_values(["numid", "mjd"], kind="stable")
    ids, start = np.unique(ph.numid.values, return_index=True)
    end = np.append(start[1:], len(ph))
    t, m, e = ph.mjd.values, ph.mag.values, ph.err.values
    return {i: (t[a:b], m[a:b], e[a:b]) for i, a, b in zip(ids, start, end)}


def fold(t, period):
    """Eqs. 8-9: phase = (t mod T) / T, in [0, 1)."""
    return np.mod(t, period) / period


def zscore_keep(m, ws=20, z=3.0, passes=2):
    """Sliding-window z-score filter on a phase-sorted curve (window 20, drop z > 3, two passes)."""
    keep = np.ones(len(m), bool)
    for _ in range(passes):
        s = pd.Series(m[keep])
        mu = s.rolling(ws, center=True, min_periods=2).mean().values
        sd = s.rolling(ws, center=True, min_periods=2).std().values
        with np.errstate(invalid="ignore", divide="ignore"):
            bad = np.abs(s.values - mu) / sd > z
        idx = np.flatnonzero(keep)
        keep[idx[bad]] = False
    return keep


def preprocess(data_dir, L=100, class_range_k=3.0, seed=0, log=print):
    """Returns a dict of arrays (one row per kept star) plus a per-class count table."""
    data_dir = Path(data_dir)
    cat = read_catalog(download(CATALOG_URL, data_dir / "CatalinaVars.tbl"))
    log(f"catalog: {len(cat):,} stars in the 6 classes")
    phot = read_photometry(download(PHOT_URL, data_dir / "AllVar.phot.gz"), cat.numid.values)
    log(f"photometry found for {len(phot):,} of them")

    # 1-2. fold, sort by phase, sliding z-score filter
    curves = []
    for r in cat.itertuples():
        if r.numid not in phot:
            continue
        t, m, e = phot[r.numid]
        ph = fold(t, r.period)
        o = np.argsort(ph, kind="stable")
        ph, m, e = ph[o], m[o], e[o]
        k = zscore_keep(m)
        curves.append((r.numid, r.cls, r.period, ph[k], m[k], e[k]))

    # 2b. class-range filter: drop a curve if > 90% of its magnitudes lie outside median +- k*std of its class
    stats = {}
    for c in CLASSES:
        allm = np.concatenate([cv[4] for cv in curves if cv[1] == c])
        stats[c] = (np.median(allm), np.std(allm))
    kept, n_range, n_short = [], 0, 0
    for cv in curves:
        med, sd = stats[cv[1]]
        if np.mean(np.abs(cv[4] - med) > class_range_k * sd) > 0.9:
            n_range += 1
        elif len(cv[4]) < L:       # 3. need at least L points to sub-sample
            n_short += 1
        else:
            kept.append(cv)
    log(f"dropped {n_range} by the class-range filter and {n_short} with fewer than {L} points")

    # 4-5. random sub-sampling to L points (kept in phase order), then median centering
    rng = np.random.default_rng(seed)
    n = len(kept)
    out = dict(numid=np.zeros(n, np.int64), y=np.zeros(n, np.int64), period=np.zeros(n),
               phase=np.zeros((n, L), np.float32), mag=np.zeros((n, L), np.float32),
               err=np.zeros((n, L), np.float32), amp=np.zeros(n, np.float32))
    for i, (numid, c, period, ph, m, e) in enumerate(kept):
        s = np.sort(rng.choice(len(m), L, replace=False))
        x = m[s] - np.median(m[s])
        out["numid"][i], out["y"][i], out["period"][i] = numid, CLASSES.index(c), period
        out["phase"][i], out["mag"][i], out["err"][i] = ph[s], x, e[s]
        out["amp"][i] = (x.max() - x.min()) / 2          # half peak-to-peak of the L points

    # split (Sec. 4.1): val and test each get 20% of the smallest class, for every class
    counts = np.bincount(out["y"], minlength=len(CLASSES))
    n_hold = int(0.2 * counts.min())
    split = np.full(n, TRAIN, np.int8)
    for c in range(len(CLASSES)):
        idx = rng.permutation(np.flatnonzero(out["y"] == c))
        split[idx[:n_hold]] = VAL
        split[idx[n_hold:2 * n_hold]] = TEST
    out["split"] = split
    out["classes"] = np.array(CLASSES)
    table = pd.DataFrame({"class": CLASSES, "ours": counts,
                          "paper": [PAPER_COUNTS[c] for c in CLASSES]})
    table["ours/paper"] = (table.ours / table.paper).round(3)
    table["train"] = [int(((out["y"] == c) & (split == TRAIN)).sum()) for c in range(len(CLASSES))]
    table["val"] = n_hold
    table["test"] = n_hold
    return out, table
