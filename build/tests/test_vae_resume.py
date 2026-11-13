"""A VAE job stopped by the time guard after epoch 1 and resumed must equal an uninterrupted run.
    python build/tests/test_vae_resume.py <folder with the Zenodo training file>
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astra import pelsvae, run_vae  # noqa: E402
from astra.common import WORK, Deadline, Logger  # noqa: E402

lcs, meta = pelsvae.load_raw(Path(sys.argv[1]))
rows = np.sort(np.random.default_rng(0).choice(len(meta), 3000, replace=False))     # all 8 classes
ds = pelsvae.make_dataset(pelsvae.normalize_each(lcs[rows]), meta.iloc[rows].reset_index(drop=True), "PTA")
X, Y, P = (torch.from_numpy(ds[k]) for k in ("x", "y", "phy"))
tr, va = pelsvae.split_indices(len(Y))
tr, va = tr[:512], va[:128]
c = dict(run_vae.HP, **run_vae.JOBS["tcn_phys"], epochs=3)


class StopAfterEpoch(Deadline):
    def __init__(self, out, n):
        super().__init__(1e12)
        self.out, self.n = out, n

    def passed(self, reserve_s=0.0):
        p = self.out / "history.json"
        return p.exists() and len(__import__("json").loads(p.read_text())) >= self.n and not (self.out / "resumed").exists()


root = WORK / "vae_resume_test"
shutil.rmtree(root, ignore_errors=True)
a, b = root / "interrupted", root / "straight"
a.mkdir(parents=True); b.mkdir(parents=True)
assert run_vae.run_job(c, X, Y, P, tr, va, a, StopAfterEpoch(a, 1), "cpu", Logger(a / "log.txt")) == "paused"
(a / "resumed").touch()
assert run_vae.run_job(c, X, Y, P, tr, va, a, Deadline(1e12), "cpu", Logger(a / "log.txt")) == "done"
assert run_vae.run_job(c, X, Y, P, tr, va, b, Deadline(1e12), "cpu", Logger(b / "log.txt")) == "done"
wa, wb = torch.load(a / "model.pt"), torch.load(b / "model.pt")
assert all(torch.equal(wa[k], wb[k]) for k in wa), "resumed weights differ"
la, lb = np.load(a / "latent.npz"), np.load(b / "latent.npz")
assert np.array_equal(la["mu"], lb["mu"])
h = __import__("json").loads((a / "history.json").read_text())
print("epochs:", [(x["epoch"], round(x["lr"], 6), round(x["val_loss"], 5)) for x in h])
print("VAE RESUME TEST PASSED (interrupted + resumed == uninterrupted, bit for bit)")
