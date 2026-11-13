"""Data for the ASTRA benchmark: the pre-processed light curves of the GAN notebook, split over stars.

What stays as in Garcia-Jara et al. (2022): the curves (L magnitudes at L sorted phases, median-centred), and
what a generator is given, the condition (phases, amplitude, class) of a real star.

What changes for the benchmark (ours):
  - the split. The paper holds out 20% of the smallest class for every class (36 curves per class on Catalina),
    and on ZTF it splits over curves, so the g and the r curve of a star can sit on both sides. Here the split
    is over STARS and stratified by class: 70% train, 10% validation, 20% test. Curves that share class and
    period are treated as one star (ZTF gives a star seen in two fields two object ids, with the same
    catalogue period), so no star is on two sides;
  - the target. Every model generates the amplitude-normalised curve u = magnitude / amplitude, which lies in
    [-2, 2] for every class, and is told the amplitude as log10(amplitude), standardised. A generated curve is
    turned back into magnitudes by multiplying with the amplitude of its condition. rep = "raw" gives the
    paper's own target instead (magnitudes, amplitude in mag), for one control run of the GAN.
"""
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from .sampling import ClassResampler

TRAIN, VAL, TEST = 0, 1, 2
FRACTIONS = (0.7, 0.1, 0.2)          # train, validation, test (of the stars of every class)
RARE_BELOW = 1000                    # a class with fewer training curves than this is in the "rare tier"
LATENT = {100: 16, 40: 8}            # noise / latent size per curve length: Catalina, ZTF (Sec. 4.2 of the paper)
G_PARAMS = {100: 3_990_785, 40: 3_708_161}   # parameters of the paper's generator; the other families match it


def star_ids(y, period):
    """One id per (class, period): the curves of one star, whatever band or object id they come from."""
    key = np.stack([y.astype(np.float64), period.astype(np.float64)], 1)
    return np.unique(key, axis=0, return_inverse=True)[1].reshape(-1)


def star_split(y, star, seed, fractions=FRACTIONS):
    """Stratified split over stars: every class keeps the same train/val/test shares, a star is on one side."""
    rng = np.random.default_rng(1000 + seed)
    stars, first = np.unique(star, return_index=True)
    ys = y[first]
    side = np.full(len(stars), TRAIN, np.int8)
    for c in np.unique(ys):
        idx = rng.permutation(np.flatnonzero(ys == c))
        n_val, n_test = int(round(fractions[1] * len(idx))), int(round(fractions[2] * len(idx)))
        side[idx[:n_val]] = VAL
        side[idx[n_val:n_val + n_test]] = TEST
    return side[np.searchsorted(stars, star)]


def split_table(npz_path, seed):
    """Curves per class and side, for the notebook's data section."""
    z = np.load(npz_path)
    split = star_split(z["y"], star_ids(z["y"], z["period"]), seed)
    n = len(z["classes"])
    return {name: np.bincount(z["y"][split == s], minlength=n) for name, s in (("train", TRAIN), ("val", VAL),
                                                                               ("test", TEST))}


def load(npz_path, seed, device, rep="unit"):
    """Everything a job needs, as tensors on `device`. Index arrays always refer to rows of the file."""
    z = np.load(npz_path)
    y = z["y"].astype(np.int64)
    star = star_ids(y, z["period"])
    split = star_split(y, star, seed)
    n_classes = len(z["classes"])
    t = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dt)
    data = {"x": t(z["mag"]), "ph": t(z["phase"]), "amp": t(z["amp"]), "y": t(y, torch.long),
            "n_classes": n_classes, "classes": [str(c) for c in z["classes"]], "L": z["mag"].shape[1],
            "y_np": y, "split": split, "star": star, "seed": seed, "rep": rep, "device": device}
    data["onehot"] = F.one_hot(data["y"], n_classes).float()
    for name, s in (("train", TRAIN), ("val", VAL), ("test", TEST)):
        idx = np.flatnonzero(split == s)
        data[f"{name}_idx_np"] = idx
        data[f"{name}_idx"] = torch.from_numpy(idx).to(device)
    tr = data["train_idx"]
    assert (data["amp"] > 0).all(), "every curve needs a positive amplitude"
    if rep == "unit":                 # u = x / amplitude in [-2, 2]; condition = standardised log-amplitude
        la = torch.log10(data["amp"])
        data["u"] = data["x"] / data["amp"][:, None]
        data["camp"] = (la - la[tr].mean()) / la[tr].std()
        data["mag_scale"] = data["amp"].clone()
        data["out_scale"] = 2.0
    elif rep == "raw":                # the paper's target: magnitudes, amplitude in mag, tanh scale = largest |mag|
        data["u"], data["camp"] = data["x"], data["amp"]
        data["mag_scale"] = torch.ones_like(data["amp"])
        data["out_scale"] = float(data["x"][tr].abs().max())
    else:
        raise ValueError(rep)
    counts = np.bincount(y[split == TRAIN], minlength=n_classes)
    data["train_counts"] = counts
    data["rare"] = counts < RARE_BELOW
    h = hashlib.md5()
    for a in (y, split, z["mag"], z["phase"]):
        h.update(np.ascontiguousarray(a).tobytes())
    data["hash"] = h.hexdigest()[:12]            # two notebooks may pair their results only if this agrees
    return data


def balanced_idx(data, n, seed, gamma=1.0, side="train"):
    """n row indices of one side, drawn by the resampling block (gamma = 1: every class equally often)."""
    pool = data[f"{side}_idx_np"]
    rs = ClassResampler(data["y_np"][pool], gamma, seed=seed, n_classes=data["n_classes"])
    return data[f"{side}_idx"][torch.from_numpy(rs.sample(n)).to(data["device"])]


def subset(data, idx, x=None):
    """A set of curves as the classifier wants it: magnitudes x, phases ph, labels y."""
    return {"x": data["x"][idx] if x is None else x, "ph": data["ph"][idx], "y": data["y"][idx]}
