# ---- 1 · Get the pre-processed files and summarise them ----
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from astra import catalina, bench_data, data_audit as audit
from astra.common import download, find_input

pd.set_option("display.width", 250)
Z_ALL, NPZ = {}, {}
for name in DATASETS:
    path = WORK / "data" / f"{name}_L{L[name]}.npz"
    if not path.exists():
        ready = find_input(path.name)
        if ready:
            shutil.copy2(ready, path)
        elif name == "catalina":                         # download + pre-process, about 3 minutes
            out, _ = catalina.preprocess(WORK / "data", L=L[name], class_range_k=CLASS_RANGE_K, seed=DATA_SEED)
            np.savez_compressed(path, **out)
        else:
            print(f"({path.name} is not attached: the {name} parts are skipped)")
            continue
    Z_ALL[name], NPZ[name] = np.load(path), path

rows = []
for name, Z in Z_ALL.items():
    t = audit.class_table(Z, SEED)
    rows.append({"dataset": name, "curves": len(Z["y"]), "stars": int(t.stars.sum()), "classes": len(t),
                 "points per curve": Z["mag"].shape[1], "largest class": f"{t['class'][t.curves.idxmax()]} ({t.curves.max():,})",
                 "smallest class": f"{t['class'][t.curves.idxmin()]} ({t.curves.min():,})",
                 "largest / smallest": round(t.curves.max() / t.curves.min()),
                 "rare tier (< 1,000 training curves)": ", ".join(t["class"][t["rare tier"]]),
                 "median amplitude (mag)": round(float(np.median(Z["amp"])), 3),
                 "median period (d)": round(float(np.median(Z["period"])), 3)})
show("The datasets", pd.DataFrame(rows), "datasets_summary.csv")
