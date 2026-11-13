"""A GAN run interrupted by the time guard and resumed must end exactly where an uninterrupted run ends.
    python build/tests/test_gan_resume.py      (needs $ASTRA_WORK/data/catalina_L100.npz from test_gan_units)
"""
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astra import classifier, gan  # noqa: E402
from astra.common import WORK, Deadline, load_json  # noqa: E402
from astra.run_gan import GAN, SMOKE  # noqa: E402

data = gan.load_data(WORK / "data" / "catalina_L100.npz", "cpu")
tr, va = gan.subset(data, data["train_idx"]), gan.subset(data, data["val_idx"])
feat, _ = classifier.train_feature_cnn(tr, va, data["n_classes"], max_epochs=1, device="cpu")
cfg = dict(GAN, gamma=0.25, seed=0, **SMOKE)
cfg.update(iters=60, ckpt_every=20, fid_batch_size=200)


class StopAt(Deadline):
    """Pretends the session time is up once the run reaches iteration `n` (only the first time)."""

    def __init__(self, run_ref, n):
        super().__init__(1e12)
        self.ref, self.n, self.fired = run_ref, n, False

    def passed(self, reserve_s=0.0):
        if not self.fired and self.ref[0] is not None and self.ref[0].it >= self.n:
            self.fired = True
            return True
        return False


root = WORK / "resume_test"
shutil.rmtree(root, ignore_errors=True)
ref = [None]
r1 = gan.GanRun(cfg, data, feat, root / "interrupted", StopAt(ref, 30), "cpu")
ref[0] = r1
assert r1.run(reserve_s=0) == "paused" and r1.it == 30
r2 = gan.GanRun(cfg, data, feat, root / "interrupted", Deadline(1e12), "cpu")
assert r2.run(reserve_s=0) == "done" and r2.it == 60
r3 = gan.GanRun(cfg, data, feat, root / "straight", Deadline(1e12), "cpu")
assert r3.run(reserve_s=0) == "done"
for a, b in zip(r2.G_ema.state_dict().values(), r3.G_ema.state_dict().values()):
    assert torch.equal(a, b), "resumed run differs from the uninterrupted run"
m2, m3 = load_json(root / "interrupted" / "metrics.json"), load_json(root / "straight" / "metrics.json")
assert [m["it"] for m in m2] == [20, 40, 60] and [round(m["fid"], 6) for m in m2] == [round(m["fid"], 6) for m in m3]
f2, f3 = load_json(root / "interrupted" / "final.json"), load_json(root / "straight" / "final.json")
assert f2["g_score"]["test_acc"] == f3["g_score"]["test_acc"]
print("final:", {k: (v["checkpoint"], v["test_acc"]) for k, v in f2.items()})
print("RESUME TEST PASSED (interrupted + resumed == uninterrupted, bit for bit)")
