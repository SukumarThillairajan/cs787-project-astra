# ---- Download (about 1 minute) and check against their Table 1 ----
import socket
from astra import common, pelsvae

have = (WORK / "data" / pelsvae.TRAIN_FILE).exists() or common.find_input(pelsvae.TRAIN_FILE) is not None
print("training file:", "found among the inputs" if have else "not attached, downloading 0.3 GB from Zenodo", flush=True)
socket.setdefaulttimeout(120)            # a stalled download raises and is retried; it cannot hang the session
lcs_raw, meta = pelsvae.load_raw(WORK / "data")
socket.setdefaulttimeout(None)
counts = pelsvae.class_counts(meta)
counts["paper: validated"] = [71, 7121, 42, 9505, 2269, 4349, 9322, 436]        # Table 1, column 4
counts["paper: augmented"] = [5000, 10045, 5090, 10000, 10365, 10044, 10169, 5047]   # Table 1, column 5
counts.to_csv(TABLES / "table1_class_counts.csv")
print(counts.to_string())
print(f"\n{len(meta):,} curves x {lcs_raw.shape[1]} points x (phase, mag, err) | "
      f"rows with period, T_eff and abs. G mag: {meta.dropna(subset=['Period', 'teff_val', 'abs_Gmag']).shape[0]:,} "
      f"({meta.dropna(subset=['Period', 'teff_val', 'abs_Gmag']).OGLE_id.nunique():,} unique stars)")

lcs = pelsvae.normalize_each(lcs_raw)
assert lcs[..., :2].min() >= 1e-4 - 1e-7 and lcs[..., :2].max() <= 0.9999 + 1e-6
classes = sorted(meta.Type.unique())
fig, ax = plt.subplots(2, 8, figsize=(22, 4.5), sharex=True)
rng = np.random.default_rng(0)
for k, c in enumerate(classes):
    i = rng.choice(np.flatnonzero(meta.Type.values == c))
    ax[0, k].plot(lcs_raw[i, :, 0], lcs_raw[i, :, 1], ".", ms=2); ax[0, k].invert_yaxis()
    ax[0, k].set_title(f"{c} (P = {meta.Period.values[i]:.3g} d)", fontsize=9)
    ax[1, k].plot(lcs[i, :, 0], lcs[i, :, 1], ".", ms=2, c="C1"); ax[1, k].invert_yaxis()
ax[0, 0].set_ylabel("I mag"); ax[1, 0].set_ylabel("scaled mag (model input)")
fig.supxlabel("phase (two cycles)"); fig.tight_layout(); fig.savefig(FIGS / "data_examples.png", dpi=110); plt.show()
del lcs_raw
