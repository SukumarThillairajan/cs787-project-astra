"""Catalogue periods for the 85,809 periodic stars of ALeRCE's labeled set.

ALeRCE's Multiband_period is fitted with one harmonic, so it is half the orbital period for most eclipsing
binaries, and an alias for about a fifth of the pulsators. The catalogues the labels come from give the period
their authors measured. This script collects it, per star, from the catalogue named in the labeled set's
`source` column, and from the AAVSO VSX (which ingests ASAS-SN, Catalina, LINEAR and Gaia variables) otherwise:
  CRTSnorth  -> CatalinaVars.tbl (Drake et al. 2014), joined on the numerical id
  CRTSsouth  -> VizieR J/MNRAS/469/3688/table4 (Drake et al. 2017)
  LINEAR     -> VizieR J/AJ/146/101/variables (Palaversa et al. 2013)
  GAIADR2VS  -> Gaia DR2 vari_rrlyrae (I/345/rrlyrae: Pf, else P1O) and vari_long_period_variable (I/345/lpv: 1/Freq)
  ASASSN, and every star still without a period -> VSX (B/vsx/vsx)
The VizieR tables are matched by position with the CDS XMatch service (nearest source within 2-3 arcsec).

    python build/ztf/catalogue_periods.py data/ztf

Writes <dir>/catalogue_periods.csv: oid, classALeRCE, source, p_src (label's own catalogue), p_vsx, p_cat
(= p_src, else p_vsx), vsx_type. The cross-match answers are cached in <dir>/xmatch/.
"""
import io
import sys
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

XMATCH = "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync"
CATALINA = "http://nesssi.cacr.caltech.edu/DataRelease/CatalinaVars.tbl"
TABLES = {"rrlyrae": ("I/345/rrlyrae", 2), "lpv": ("I/345/lpv", 2), "vsx": ("B/vsx/vsx", 3),
          "crtssouth": ("J/MNRAS/469/3688/table4", 3), "linear": ("J/AJ/146/101/variables", 3)}
CLASSES = ["E", "RRL", "LPV", "DSCT", "CEP"]


def xmatch(positions_csv, table, radius, out):
    """Nearest VizieR source within `radius` arcsec of every (ra, dec) row; cached in `out`."""
    if not out.exists():
        fields = {"request": "xmatch", "distMaxArcsec": str(radius), "RESPONSEFORMAT": "csv", "colRA1": "ra",
                  "colDec1": "dec", "cat2": f"vizier:{table}"}
        b = uuid.uuid4().hex
        body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items())
        body += (f'--{b}\r\nContent-Disposition: form-data; name="cat1"; filename="pos.csv"\r\n'
                 "Content-Type: text/csv\r\n\r\n").encode() + positions_csv + f"\r\n--{b}--\r\n".encode()
        req = urllib.request.Request(XMATCH, data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
        out.write_bytes(urllib.request.urlopen(req, timeout=1800).read())
    return pd.read_csv(out, low_memory=False).sort_values("angDist").drop_duplicates("oid").set_index("oid")


def main():
    d = Path(sys.argv[1])
    (d / "xmatch").mkdir(exist_ok=True)
    lab = pd.read_csv(d / "labeled_set_lc_classifier_SanchezSaez_2020.csv")
    lab = lab[lab.classALeRCE.isin(CLASSES)].drop_duplicates("oid").set_index("oid")
    pos = lab[["ra", "dec"]].to_csv().encode()
    x = {k: xmatch(pos, t, r, d / "xmatch" / f"{k}.csv") for k, (t, r) in TABLES.items()}

    f = d / "xmatch" / "CatalinaVars.tbl"
    if not f.exists():
        f.write_bytes(urllib.request.urlopen(CATALINA, timeout=600).read())
    rows = [ln.split() for ln in f.read_text().splitlines() if not ln.startswith("#")]
    cat = pd.DataFrame([(int(r[1]), r[5]) for r in rows if len(r) >= 9], columns=["nid", "p"]).drop_duplicates("nid")
    cat = pd.to_numeric(cat.set_index("nid")["p"], errors="coerce")           # "\nodata" for the aperiodic type 17

    src, cls, p = lab.source, lab.classALeRCE, pd.Series(np.nan, lab.index)
    m = src == "CRTSnorth"
    p[m] = lab[m].id_source.astype(float).astype("int64").map(cat).values
    for name, col, m in (("crtssouth", "Per", src == "CRTSsouth"), ("linear", "Period", src == "LINEAR")):
        p[m] = pd.to_numeric(x[name][col], errors="coerce").reindex(lab.index[m]).values
    m = (src == "GAIADR2VS") & (cls == "RRL")
    p[m] = x["rrlyrae"]["Pf"].fillna(x["rrlyrae"]["P1O"]).reindex(lab.index[m]).values
    m = (src == "GAIADR2VS") & (cls == "LPV")
    p[m] = (1 / x["lpv"]["Freq"]).reindex(lab.index[m]).values
    vsx = x["vsx"].reindex(lab.index)
    out = pd.DataFrame({"classALeRCE": cls, "source": src, "p_src": p.where(p > 0),
                        "p_vsx": pd.to_numeric(vsx["Period"], errors="coerce").where(lambda s: s > 0),
                        "vsx_type": vsx["Type"]})
    out["p_cat"] = out.p_src.fillna(out.p_vsx)
    out[["classALeRCE", "source", "p_src", "p_vsx", "p_cat", "vsx_type"]].to_csv(d / "catalogue_periods.csv")

    print("stars with a catalogue period:", out.groupby("classALeRCE").p_cat.apply(lambda s: int(s.notna().sum())).reindex(CLASSES).to_dict())
    raw = d / "ztf_periodic_raw.npz"
    if raw.exists():
        z = np.load(raw)
        r = pd.Series(z["period"], index=z["oid"]).reindex(out.index) / out.p_cat
        print("ALeRCE lc_classifier_1.2.1-P period against the catalogue period (within 1%):")
        for c in CLASSES:
            q = r[(cls == c) & r.notna()]
            share = {k: float(np.mean(np.abs(q / v - 1) < 0.01)) for k, v in (("equal", 1), ("half", 0.5), ("double", 2))}
            print(f"  {c:5s} n = {len(q):6d}  " + "  ".join(f"{k} {v:.3f}" for k, v in share.items()) + f"  other {1 - sum(share.values()):.3f}")


if __name__ == "__main__":
    main()
