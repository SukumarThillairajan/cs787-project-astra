# ---- Get the pre-processed light curves (made once; later sessions reuse the saved arrays) ----
import matplotlib.pyplot as plt
import pandas as pd
from astra import catalina, ztf
from astra.common import find_input

NPZ = WORK / "data" / f"{DATASET}_L{L}.npz"
if not NPZ.exists() and DATASET == "catalina":           # download + pre-process, about 3 minutes
    out, _ = catalina.preprocess(WORK / "data", L=L, class_range_k=CLASS_RANGE_K, seed=DATA_SEED)
    np.savez_compressed(NPZ, **out)
    (WORK / "data" / "AllVar.phot.gz").unlink(missing_ok=True)      # raw photometry no longer needed
if not NPZ.exists() and DATASET == "ztf":                # attached: the ready file, or the raw download
    ready, raw = find_input(NPZ.name), find_input("ztf_periodic_raw.npz")
    assert ready or raw, "ZTF: use Add Input to attach the dataset with ztf_L40.npz (see the data section above)"
    if ready:
        shutil.copy2(ready, NPZ)
    else:
        out, _ = ztf.preprocess(raw, L=L, seed=DATA_SEED)
        np.savez_compressed(NPZ, **out)
Z = np.load(NPZ)
CLASSES = [str(c) for c in Z["classes"]]
paper = (catalina if DATASET == "catalina" else ztf).PAPER_COUNTS
table = pd.DataFrame({"class": CLASSES, "ours": np.bincount(Z["y"]), "paper": [paper[c] for c in CLASSES]})
table["ours/paper"] = (table.ours / table.paper).round(3)
for name, s in (("train", 0), ("val", 1), ("test", 2)):
    table[name] = np.bincount(Z["y"][Z["split"] == s], minlength=len(CLASSES))
table.to_csv(TABLES / "table2_class_counts.csv", index=False)
print(table.to_string(index=False))
print(f"\n{len(Z['y']):,} light curves of {L} points | train {np.sum(Z['split'] == 0):,}, "
      f"val {np.sum(Z['split'] == 1)}, test {np.sum(Z['split'] == 2)}")

# a few pre-processed curves per class (like Fig. 1b); magnitudes grow downwards, so the y axis is flipped
fig, ax = plt.subplots(2, len(CLASSES), figsize=(3 * len(CLASSES), 5), sharex=True)
rng = np.random.default_rng(0)
for c, name in enumerate(CLASSES):
    for r, i in enumerate(rng.choice(np.flatnonzero(Z["y"] == c), 2, replace=False)):
        ax[r, c].plot(Z["phase"][i], Z["mag"][i], ".", ms=3)
        ax[r, c].invert_yaxis()
        ax[r, c].set_title(f"{name} (P = {Z['period'][i]:.3g} d)", fontsize=9)
fig.supxlabel("phase"); fig.supylabel("centred magnitude")
fig.tight_layout(); fig.savefig(FIGS / "data_examples.png", dpi=110); plt.show()
