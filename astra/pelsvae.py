"""PELS-VAE data (Martinez-Palomera, Bloom & Abrahams 2020), loaded exactly as their released code does
(github.com/jorgemarpa/PELS-VAE, src/datasets.py and vae_main.py).

The Zenodo training file holds 65,760 OGLE-III I-band light curves of 8 classes: 33,124 real stars
(phase-folded, 300 random points) plus 32,636 augmented copies (magnitudes resampled from N(m, err),
random phase shift, re-sampled). Every curve is shown over 2 phase cycles, so it has 600 points.
Each curve is min-max scaled on its own to [0.0001, 0.9999] (phase and magnitude; errors by the
magnitude range). Their split: a random 80/20 train/validation split with seed 13.
"""
import gzip

import numpy as np
import pandas as pd

from .common import download

ZENODO = "https://zenodo.org/records/3820679/files/"
TRAIN_FILE = "OGLE3_lcs_I_meta_snr5_augmented_folded_trim600.npy.gz"
PHYS = {"P": "Period", "T": "teff_val", "M": "[Fe/H]_J95", "C": "bp_rp", "A": "abs_Gmag", "R": "radius_val",
        "L": "lum_val"}           # their --phy letters, in their order


def load_raw(data_dir):
    path = download(ZENODO + TRAIN_FILE + "?download=1", data_dir / TRAIN_FILE)
    with gzip.open(path, "rb") as f:
        d = np.load(f, allow_pickle=True).item()
    return d["lcs"].astype(np.float32), d["meta"].reset_index(drop=True)


def normalize_each(lcs, lo=0.0001, hi=0.9999):
    """Their utils.normalize_each(norm_time=True, scale_to=[.0001, .9999]), vectorised."""
    out = np.empty_like(lcs)
    t, m, e = lcs[..., 0], lcs[..., 1], lcs[..., 2]
    tr = t.max(1, keepdims=True) - t.min(1, keepdims=True)
    mr = m.max(1, keepdims=True) - m.min(1, keepdims=True)
    out[..., 0] = (t - t.min(1, keepdims=True)) / tr * (hi - lo) + lo
    out[..., 1] = (m - m.min(1, keepdims=True)) / mr * (hi - lo) + lo
    out[..., 2] = e / mr * (hi - lo)
    return out


def make_dataset(lcs_norm, meta, phy=""):
    """Their Astro_lightcurves(...) + remove_nan(): one-hot labels (alphabetical classes), min-max scaled
    physical parameters (scaler fitted on all rows, NaNs ignored), rows with missing parameters dropped."""
    classes = sorted(meta.Type.unique())
    y = meta.Type.map({c: i for i, c in enumerate(classes)}).values.astype(np.int64)
    names = [PHYS[k] for k in "PTMCARL" if k in phy.upper()]
    cols = names if names else ["Period"]
    p = meta[cols].values.astype(np.float32)
    lo, hi = np.nanmin(p, 0), np.nanmax(p, 0)
    p = (p - lo) / (hi - lo)
    keep = ~np.isnan(p).any(1)
    return {"x": lcs_norm[keep], "y": y[keep], "phy": p[keep] if names else np.zeros((keep.sum(), 0), np.float32),
            "rows": np.flatnonzero(keep), "classes": classes, "phy_names": names}


def split_indices(n, seed=13, test_split=0.2):
    """Their get_dataloader(): np.random.seed(seed); shuffle list(range(n)); first 20% = validation."""
    np.random.seed(seed)
    idx = list(range(n))
    np.random.shuffle(idx)
    k = int(np.floor(test_split * n))
    return np.array(idx[k:]), np.array(idx[:k])


def class_counts(meta):
    real = meta.drop_duplicates("OGLE_id")
    return pd.DataFrame({"curves (with augmented)": meta.Type.value_counts().sort_index(),
                         "unique stars": real.Type.value_counts().sort_index()})
