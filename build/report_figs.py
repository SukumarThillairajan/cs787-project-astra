"""Figures for the report and slides that compare_bench.py does not make, from the final report zips.

    python build/report_figs.py <out_dir>

Reads Outputs/Benchmark/final/*/report_*.zip (unzipped into a temporary folder) and writes:
  fig_timeline.png  validation TSTR and rare-tier copy share against training time, mean over splits 0-4 per family
"""
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from astra import bench_report as rep  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
tmp = Path(tempfile.mkdtemp())
for z in sorted((ROOT / "Outputs/Benchmark/final").glob("*/report_*.zip")):
    zipfile.ZipFile(z).extractall(tmp / z.parent.name)
res = rep.collect([tmp], "results_catalina")

COLORS = {"gan": "tab:blue", "vae": "tab:orange", "ddpm": "tab:green", "fm": "tab:red"}
NAMES = {"gan": "GAN", "vae": "VAE", "ddpm": "DDPM", "fm": "flow matching"}
plt.rcParams.update({"font.size": 12})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
for fam in ("gan", "vae", "ddpm", "fm"):
    runs = [r for j, r in res.items() if r.get("family") == fam]
    curves = {}                                     # minute -> values over splits
    for r in runs:
        for m in r["metrics"]:
            t = round(60 * m["solo_h"], 1)
            curves.setdefault(t, []).append((100 * m["val_bal_acc"], 100 * rep.rare_mean(r, m["copy_share"])))
    ts = sorted(t for t, v in curves.items() if len(v) == len(runs))   # checkpoints every split reached
    acc = np.array([[v[0] for v in curves[t]] for t in ts])
    cop = np.array([[v[1] for v in curves[t]] for t in ts])
    for ax, y in ((a1, acc), (a2, cop)):
        ax.plot(ts, y.mean(1), "o-", ms=3, color=COLORS[fam], label=f"{NAMES[fam]} ({len(runs)} splits)")
        ax.fill_between(ts, y.min(1), y.max(1), color=COLORS[fam], alpha=0.15)
    sel = [60 * r["selected_solo_h"] for r in runs]
    a1.axvline(np.median(sel), color=COLORS[fam], ls=":", lw=1)
for ax in (a1, a2):
    ax.set_xscale("log")
    ax.set_xlabel("minutes of training on one T4")
    ax.grid(alpha=0.3)
a1.set_ylabel("validation TSTR balanced acc. %")
a1.set_ylim(15, 90)
a1.set_title("score (dotted: median selected checkpoint)")
a2.set_ylabel("rare-class curves copied %")
a2.axhline(5, color="grey", lw=1)
a2.set_title("memorisation (grey: level of real curves)")
a2.legend(loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig(out / "fig_timeline.png", dpi=200)
print("wrote", out / "fig_timeline.png", {f: sum(r.get("family") == f for r in res.values()) for f in COLORS})
