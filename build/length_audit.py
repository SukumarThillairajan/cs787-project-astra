"""How many Catalina stars does the fixed-length (L = 100) pipeline of Garcia-Jara et al. throw away, per class?

Runs the same steps as astra.catalina.preprocess up to the length filter (fold, sliding z-score, class-range
filter with the benchmark's k = 2.0) and reports, per class, the number of points per curve and how many curves
have fewer than L points (the paper and our benchmark drop those).

Usage: python build/length_audit.py <dir with CatalinaVars.tbl + AllVar.phot.gz> <out dir> [k]
Writes <out>/length_audit.csv and <out>/length_audit.png. CPU, about 2 min.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra import catalina as C  # noqa: E402


def main(raw, out, k=2.0):
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cat = C.read_catalog(raw / "CatalinaVars.tbl")
    phot = C.read_photometry(raw / "AllVar.phot.gz", cat.numid.values)
    curves = []
    for r in cat.itertuples():
        if r.numid not in phot:
            continue
        t, m, e = phot[r.numid]
        ph = C.fold(t, r.period)
        o = np.argsort(ph, kind="stable")
        m = m[o]
        curves.append((r.cls, m[C.zscore_keep(m)]))
    stats = {}
    for c in C.CLASSES:
        allm = np.concatenate([m for cls, m in curves if cls == c])
        stats[c] = (np.median(allm), np.std(allm))
    lens = {c: [] for c in C.CLASSES}
    for cls, m in curves:
        med, sd = stats[cls]
        if np.mean(np.abs(m - med) > k * sd) > 0.9:
            continue
        lens[cls].append(len(m))
    rows = []
    for c in C.CLASSES:
        n = np.array(lens[c])
        kept = n[n >= 100]
        rows.append(dict(cls=c, curves=len(n), kept_L100=int((n >= 100).sum()),
                         obs_unused_pct=round(100 * (kept - 100).sum() / kept.sum(), 1),
                         dropped_lt100=int((n < 100).sum()), dropped_pct=round(100 * (n < 100).mean(), 1),
                         ge50=int((n >= 50).sum()), ge30=int((n >= 30).sum()), ge20=int((n >= 20).sum()),
                         median_len=int(np.median(n)), p10=int(np.percentile(n, 10)),
                         p90=int(np.percentile(n, 90))))
    allk = np.concatenate([np.array(lens[c]) for c in C.CLASSES])
    allk = allk[allk >= 100]
    rows.append(dict(cls="all", curves=sum(len(v) for v in lens.values()), kept_L100=len(allk),
                     obs_unused_pct=round(100 * (allk - 100).sum() / allk.sum(), 1)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "length_audit.csv", index=False)
    print(df.to_string(index=False))
    np.savez_compressed(out / "length_audit_lengths.npz", **{c.replace("/", "_"): np.array(lens[c]) for c in C.CLASSES})
    plot(lens, df, out)


def plot(lens, df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for k, c in enumerate(C.CLASSES):
        n = np.array(lens[c])
        yk = len(C.CLASSES) - 1 - k
        jit = np.random.default_rng(k).uniform(-0.28, 0.28, len(n))
        if len(n) > 1500:
            s = np.random.default_rng(k).choice(len(n), 1500, replace=False)
            nn, jj = n[s], jit[s]
        else:
            nn, jj = n, jit
        ax.scatter(nn, yk + jj, s=1.2, lw=0, c=np.where(nn < 100, "#d1495b", "#7f93a8"), alpha=0.6)
        q1, med, q3 = np.percentile(n, [25, 50, 75])
        ax.plot([q1, q3], [yk, yk], c="#22303f", lw=3, solid_capstyle="butt")
        ax.plot([med], [yk], "o", c="white", mec="#22303f", ms=4)
        r = df[df.cls == c].iloc[0]
        ax.text(760, yk, f"{r.dropped_pct:.1f}% of curves dropped;  {r.obs_unused_pct:.0f}% of points unused",
                va="center", fontsize=7)
    ax.axvspan(0, 100, color="#fbe3e6", lw=0, zorder=0)
    ax.axvline(100, c="#d1495b", lw=1, zorder=0.5)
    ax.text(104, len(C.CLASSES) - 0.05, "L = 100: shorter curves are dropped (red), "
            "longer ones are cut to 100 random points", fontsize=6.8, color="#a3283a", va="center")
    ax.set_yticks(range(len(C.CLASSES))); ax.set_yticklabels(C.CLASSES[::-1])
    ax.set_xlim(0, 750); ax.set_ylim(-0.6, len(C.CLASSES) + 0.2)
    ax.set_xlabel("observations per light curve (after the outlier filters; bar = middle 50%, dot = median)")
    fig.tight_layout()
    fig.savefig(out / "length_audit.png", dpi=220, bbox_inches="tight")
    print("wrote", out / "length_audit.csv", out / "length_audit.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2.0)
