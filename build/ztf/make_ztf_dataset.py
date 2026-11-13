"""Build data/ztf/ztf_L40.npz (the file the GAN notebook needs) from the raw download.

    python build/ztf/download_ztf.py <labeled_set.csv> data/ztf      # once, over an hour
    python build/ztf/add_label_periods.py data/ztf                   # once: the period the paper folds with
    python build/ztf/make_ztf_dataset.py data/ztf
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astra import ztf  # noqa: E402

d = Path(sys.argv[1])
out, table = ztf.preprocess(d / "ztf_periodic_raw.npz", L=40, seed=0)
np.savez_compressed(d / "ztf_L40.npz", **out)
table.to_csv(d / "ztf_table2_class_counts.csv", index=False)
print(table.to_string(index=False))
print("amplitude median per class:", {c: round(float(np.median(out["amp"][out["y"] == i])), 3) for i, c in enumerate(ztf.CLASSES)})
print("max |mag| on train:", float(np.abs(out["mag"][out["split"] == 0]).max()))
