"""Build the raw ZTF dataset of Garcia-Jara et al. (2022): ALeRCE labels + ALeRCE detections + periods.

The paper uses "the subset of periodic variable stars present in the ZTF training set created by the ALeRCE
broker". That labeled set (Sanchez-Saez et al. 2021) is on Zenodo (record 4279623) but holds only object
ids and classes. The light curves themselves are served by ALeRCE's public TAP service
(https://tap.alerce.online/tap, table ztf.detection), and the period by table ztf.feature
(name = 'Multiband_period'). This script downloads both for the 85,809 periodic objects, in resumable
batches over a few parallel connections, and writes one compact file.

    python build/ztf/download_ztf.py <labeled_set.csv> <out_dir> [n_connections]

Output: <out_dir>/ztf_periodic_raw.npz with, per detection: obj (index into oid), mjd, fid (1 = g, 2 = r),
mag (magpsf_corr), err (sigmapsf_corr), err_ext, corrected, dubious; and per object: oid, cls, period
(classifier version of the paper's time), period_new (newest classifier version).
"""
import gzip
import io
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

TAP = "https://tap.alerce.online/tap/sync"
CLASSES = ["E", "RRL", "LPV", "DSCT", "CEP"]            # the 5 periodic classes of the paper's Table 2
BATCH = 200


def tap(query, tries=8):
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query,
                                   "MAXREC": "20000000"}).encode()
    for i in range(tries):
        try:
            req = urllib.request.Request(TAP, data=data, headers={"Accept-Encoding": "gzip"})
            r = urllib.request.urlopen(req, timeout=1200)
            raw = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            return pd.read_csv(io.BytesIO(raw))
        except Exception as e:                           # server hiccup: wait and retry
            time.sleep(15 * (i + 1))
            err = e
    raise RuntimeError(f"TAP query failed: {err}")


def fetch(kind, k, oids, out):
    f = out / f"{kind}_{k:04d}.csv.gz"
    if f.exists():
        return
    inlist = ",".join(f"'{o}'" for o in oids)
    if kind == "det":
        q = ("SELECT oid, mjd, fid, magpsf_corr, sigmapsf_corr, sigmapsf_corr_ext, corrected, dubious "
             f"FROM ztf.detection WHERE oid IN ({inlist})")
    else:
        q = ('SELECT f.oid, f.version, f."value" FROM ztf.feature AS f '        # "value" is a reserved word
             f"WHERE f.name = 'Multiband_period' AND f.oid IN ({inlist})")
    df = tap(q)
    tmp = f.with_suffix(".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(f)


def main():
    lab = pd.read_csv(sys.argv[1])
    out = Path(sys.argv[2])
    ncon = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    (out / "parts").mkdir(parents=True, exist_ok=True)
    lab = lab[lab.classALeRCE.isin(CLASSES)].drop_duplicates("oid").sort_values("oid").reset_index(drop=True)
    print(f"{len(lab):,} periodic objects:", lab.classALeRCE.value_counts().to_dict(), flush=True)
    batches = [lab.oid.values[i:i + BATCH] for i in range(0, len(lab), BATCH)]
    jobs = [("per", k, b) for k, b in enumerate(batches)] + [("det", k, b) for k, b in enumerate(batches)]
    t0, done = time.time(), [0]

    def run(j):
        fetch(j[0], j[1], j[2], out / "parts")
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"{done[0]}/{len(jobs)} batches, {(time.time() - t0) / 60:.1f} min", flush=True)

    with ThreadPoolExecutor(ncon) as ex:
        list(ex.map(run, jobs))

    # merge into one compact file
    # the feature table keeps one period per classifier version. "lc_classifier_1.2.1-P" is the version of
    # Sanchez-Saez et al. (2021), the one available to the paper; the newest version is kept as a fallback.
    per = pd.concat([pd.read_csv(out / "parts" / f"per_{k:04d}.csv.gz") for k in range(len(batches))]).dropna()
    per.to_csv(out / "ztf_periods_all_versions.csv.gz", index=False, compression="gzip")
    paper = per[per.version == "lc_classifier_1.2.1-P"].drop_duplicates("oid").set_index("oid")["value"]
    new = per[~per.version.str.startswith("lc_classifier") & (per["value"] != 1.0)]
    new = new.sort_values("version").drop_duplicates("oid", keep="last").set_index("oid")["value"]
    idx = {o: i for i, o in enumerate(lab.oid.values)}
    cols = {k: [] for k in ("obj", "mjd", "fid", "mag", "err", "err_ext", "corrected", "dubious")}
    for k in range(len(batches)):
        d = pd.read_csv(out / "parts" / f"det_{k:04d}.csv.gz")
        cols["obj"].append(d.oid.map(idx).values.astype(np.int32))
        cols["mjd"].append(d.mjd.values.astype(np.float64))
        cols["fid"].append(d.fid.values.astype(np.int8))
        cols["mag"].append(d.magpsf_corr.values.astype(np.float32))
        cols["err"].append(d.sigmapsf_corr.values.astype(np.float32))
        cols["err_ext"].append(d.sigmapsf_corr_ext.values.astype(np.float32))
        cols["corrected"].append(d.corrected.astype(str).str.lower().eq("true").values)
        cols["dubious"].append(d.dubious.astype(str).str.lower().eq("true").values)
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.lexsort((cols["mjd"], cols["obj"]))
    cols = {k: v[order] for k, v in cols.items()}
    np.savez_compressed(out / "ztf_periodic_raw.npz", oid=lab.oid.values.astype("U12"),
                        cls=lab.classALeRCE.values.astype("U8"),
                        period=paper.reindex(lab.oid.values).values.astype(np.float64),
                        period_new=new.reindex(lab.oid.values).values.astype(np.float64), **cols)
    print(f"saved {len(cols['mjd']):,} detections of {len(lab):,} objects ({paper.reindex(lab.oid.values).notna().sum():,} "
          f"with the paper-era period, {new.reindex(lab.oid.values).notna().sum():,} with a newer one) "
          f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
