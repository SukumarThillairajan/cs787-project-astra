# ---- Get the pre-processed light curves, then split them over stars ----
from astra import catalina, ztf, bench_data
from astra.common import find_input

NPZ = WORK / "data" / f"{DATASET}_L{L}.npz"
if not NPZ.exists():
    ready = find_input(NPZ.name)
    if ready:
        shutil.copy2(ready, NPZ)
    elif DATASET == "catalina":                          # download + pre-process, about 3 minutes
        out, _ = catalina.preprocess(WORK / "data", L=L, class_range_k=CLASS_RANGE_K, seed=DATA_SEED)
        np.savez_compressed(NPZ, **out)
        (WORK / "data" / "AllVar.phot.gz").unlink(missing_ok=True)
    else:
        raise AssertionError("ZTF: use Add Input to attach the dataset with ztf_L40.npz")
Z = np.load(NPZ)
CLASSES = [str(c) for c in Z["classes"]]
table = pd.DataFrame({"class": CLASSES, "curves": np.bincount(Z["y"], minlength=len(CLASSES))})
for s in SEEDS:
    t = bench_data.split_table(NPZ, s)
    for side in ("train", "val", "test"):
        table[f"{side} (split {s})"] = t[side]
table["rare tier"] = bench_data.split_table(NPZ, SEEDS[0])["train"] < bench_data.RARE_BELOW
table.to_csv(TABLES / "split_counts.csv", index=False)
print(table.to_string(index=False))
d0 = bench_data.load(NPZ, SEEDS[0], "cpu")
stars = [set(d0["star"][d0[f"{s}_idx_np"]]) for s in ("train", "val", "test")]
assert not (stars[0] & stars[1]) and not (stars[0] & stars[2]) and not (stars[1] & stars[2]), "a star is on two sides"
print(f"\n{len(Z['y']):,} curves of {L} points, {len(np.unique(d0['star'])):,} stars | split {SEEDS[0]}: "
      f"train {len(d0['train_idx']):,}, val {len(d0['val_idx']):,}, test {len(d0['test_idx']):,} | no star on two sides "
      f"| data hash {d0['hash']}")

# one curve per class: magnitudes (top) and the model's target u = magnitude / amplitude (bottom)
fig, ax = plt.subplots(2, len(CLASSES), figsize=(3 * len(CLASSES), 5), sharex=True)
rng = np.random.default_rng(0)
for c, name in enumerate(CLASSES):
    i = rng.choice(np.flatnonzero(Z["y"] == c))
    ax[0, c].plot(Z["phase"][i], Z["mag"][i], ".", ms=3); ax[0, c].invert_yaxis()
    ax[0, c].set_title(f"{name} (amplitude {Z['amp'][i]:.2f} mag)", fontsize=9)
    ax[1, c].plot(Z["phase"][i], Z["mag"][i] / Z["amp"][i], ".", ms=3, c="C1"); ax[1, c].set_ylim(2.1, -2.1)
ax[0, 0].set_ylabel("centred magnitude"); ax[1, 0].set_ylabel("magnitude / amplitude")
fig.supxlabel("phase"); fig.tight_layout(); fig.savefig(FIGS / "data_examples.png", dpi=110); plt.show()
del d0
