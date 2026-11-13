"""Add the period the paper folds with to the raw ZTF download, as the array `period_label`.

Garcia-Jara et al. fold "with the period provided". Matching their plotted light curves to ALeRCE's detections
shows that this is the `period` column of ALeRCE's internal label file: the period of the catalogue the label
came from (CRTS, LINEAR, ASAS-SN), absent for most Gaia-labelled stars. The public labeled set on Zenodo has no
such column. A public copy of the internal file is ZTF_labels_v8.0.1.csv, inside data_raw.zip (1.38 GB) of
Perez-Carrasco et al. 2023 (github.com/mperezcarrasco/AnomalyALeRCE, data/get_data.sh). If <dir> does not hold
the CSV yet, this script reads only that member of the zip (about 12 MB) with HTTP range requests.

    python build/ztf/add_label_periods.py data/ztf

Appends `period_label` (days; NaN where the file has no number) to <dir>/ztf_periodic_raw.npz.
"""
import io
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DRIVE_ID = "1Wwh6TJYq4i65tG5Y4KxFq-wfwslV0pOO"
MEMBER = "data_raw/ZTF_labels_v8.0.1.csv"
UA = {"User-Agent": "Mozilla/5.0"}


class RangeFile(io.RawIOBase):
    """A remote file read in 1 MB blocks, so that zipfile fetches only the directory and one member."""

    def __init__(self, url):
        self.url, self.pos, self.blocks = url, 0, {}
        r = urllib.request.urlopen(urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"}), timeout=120)
        self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        out = b""
        while len(out) < n:
            k, o = divmod(self.pos + len(out), 1 << 20)
            if k not in self.blocks:
                rng = f"bytes={k << 20}-{min((k + 1) << 20, self.size) - 1}"
                req = urllib.request.Request(self.url, headers={**UA, "Range": rng})
                self.blocks[k] = urllib.request.urlopen(req, timeout=300).read()
            out += self.blocks[k][o:o + n - len(out)]
        self.pos += len(out)
        return out


def fetch_labels(path):
    base = f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download"
    page = urllib.request.urlopen(urllib.request.Request(base, headers=UA), timeout=60).read().decode("utf-8", "replace")
    uuid = re.search(r'name="uuid" value="([^"]+)"', page)          # the "too large to scan" confirmation form
    url = base + "&confirm=t" + (f"&uuid={uuid.group(1)}" if uuid else "")
    path.write_bytes(zipfile.ZipFile(RangeFile(url)).read(MEMBER))


def main():
    d = Path(sys.argv[1])
    csv, raw = d / "ZTF_labels_v8.0.1.csv", d / "ztf_periodic_raw.npz"
    if not csv.exists():
        fetch_labels(csv)
    lab = pd.read_csv(csv, low_memory=False).drop_duplicates("oid").set_index("oid")
    z = np.load(raw)
    assert "period_label" not in z.files, "period_label is already in the raw file"
    lab = lab.reindex(z["oid"])
    period = pd.to_numeric(lab.period, errors="coerce").where(lambda p: p > 0).values.astype(np.float64)
    cls = z["cls"]
    z.close()

    backup = raw.with_suffix(".bak")                                 # an interrupted append would break the zip
    shutil.copy2(raw, backup)
    buf = io.BytesIO()
    np.lib.format.write_array(buf, period)
    with zipfile.ZipFile(raw, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("period_label.npy", buf.getvalue())
    assert np.array_equal(np.load(raw)["period_label"], period, equal_nan=True)
    backup.unlink()

    print(f"{int(lab.classALeRCE.notna().sum()):,} of {len(period):,} stars are in the label file; with a period:")
    for c in ("E", "RRL", "LPV", "DSCT", "CEP"):
        s = cls == c
        print(f"  {c:5s} {int(np.isfinite(period[s]).sum()):6,} of {int(s.sum()):6,}")


if __name__ == "__main__":
    main()
