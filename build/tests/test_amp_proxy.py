"""Mixed precision cannot run in fp16 on a CPU, but CPU bfloat16 autocast goes through the same code:
dtype handling, the scaled gradient penalty, GradScaler stepping and its save/restore. This test forces
that path for a few classifier and GAN steps and checks everything stays finite and resumable.
    python build/tests/test_amp_proxy.py      (needs $ASTRA_WORK/data/catalina_L100.npz from test_gan_units)
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astra import classifier, gan  # noqa: E402
from astra.common import WORK, Deadline  # noqa: E402
from astra.run_gan import GAN, SMOKE  # noqa: E402

# force "AMP on" with the CPU's bfloat16
_Scaler = torch.amp.GradScaler
for mod in (classifier, gan):
    mod.amp_on = lambda amp, device: bool(amp)
    mod.autocast = lambda device, amp: torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(amp))
torch.amp.GradScaler = lambda enabled=True: _Scaler("cpu", enabled=enabled)

data = gan.load_data(WORK / "data" / "catalina_L100.npz", "cpu")
keep = torch.cat([data["train_idx"][data["y"][data["train_idx"]] == c][:200] for c in range(data["n_classes"])])
small = {k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == data["x"].shape[:1] else v) for k, v in data.items()}
small["y_np"] = data["y_np"][keep.numpy()]
n = len(keep)
small["train_idx"], small["train_idx_np"] = torch.arange(n), np.arange(n)
small["val_idx"] = small["test_idx"] = torch.arange(0, n, 5)

# classifier: bf16 result close to the full-precision one, training loss finite and decreasing
tr, va = gan.subset(small, small["train_idx"]), gan.subset(small, small["val_idx"])
ens = classifier.CNNEnsemble(3, data["n_classes"], 100).eval()
X = classifier.stack_input(tr["x"][:32], tr["ph"][:32])
p16, p32 = classifier.predict_proba(ens, X, amp=True), classifier.predict_proba(ens, X, amp=False)
assert p16.dtype == torch.float32 and (p16 - p32).abs().max() < 0.05, (p16 - p32).abs().max()
r = classifier.fit_ensembles(tr, va, va, data["n_classes"], runs=1, ens=3, max_epochs=3, patience=3, device="cpu",
                             log=lambda *a: None, amp=True)
c = r["curves"]["train_loss"]
assert np.isfinite(c).all() and c[-1] < c[0], c
print("classifier under autocast ok: train loss", np.round(c, 3))

# GAN: steps with the scaled gradient penalty, state save/load with both scalers
feat, _ = classifier.train_feature_cnn(tr, va, data["n_classes"], max_epochs=1, device="cpu", log=lambda *a: None)
cfg = dict(GAN, gamma=0.25, seed=0, **SMOKE)
out = WORK / "amp_proxy_test"
shutil.rmtree(out, ignore_errors=True)
run = gan.GanRun(cfg, small, feat, out, Deadline(1e12), "cpu")
assert run.amp and run.scalerD.is_enabled()
# why the penalty caps its scale: the scaler's initial value does not fit into fp16, the cap does
assert torch.isinf(torch.tensor(run.scalerD.get_scale()).half()) and torch.isfinite(torch.tensor(gan.GP_SCALE_MAX).half())
x_r, cond, y = run.batch(run.sampler.sample(8))
x_g = run.G(torch.randn(8, 16), *cond).detach()
gp_amp = gan.gradient_penalty(run.D, x_r, x_g, cond, run.scalerD, True)
torch.manual_seed(1); a = gan.gradient_penalty(run.D, x_r, x_g, cond, run.scalerD, True)
torch.manual_seed(1); b = gan.gradient_penalty(run.D, x_r, x_g, cond, None, False)
assert abs(a.item() - b.item()) < 0.05 * abs(b.item()) + 1e-3, (a.item(), b.item())     # same penalty, scaled or not
losses = torch.stack([run.step() for _ in range(6)])
assert torch.isfinite(losses).all() and float(run.skipped) == 0, (losses, run.skipped)
run.save_state()
run2 = gan.GanRun(cfg, small, feat, out, Deadline(1e12), "cpu")
assert run2.load_state() and run2.it == 6 and run2.scalerD.get_scale() == run.scalerD.get_scale()
assert all(p.dtype == torch.float32 for p in run.G.parameters())
print(f"GAN under autocast ok: gp scaled {a.item():.4f} vs plain {b.item():.4f}; losses D/G/W", losses[-1].tolist())
print("AMP PROXY TEST PASSED")
