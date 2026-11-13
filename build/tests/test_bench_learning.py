"""Does each family learn? (CPU, a few minutes per job.) Run from the project root:
    python build/tests/test_bench_learning.py <catalina_L100.npz or ztf_L40.npz> [names ...]
A name is a family (gan, vae, ddpm, fm: its first grid configuration) or a job (e.g. ddpm_v_lr2e-4_s0),
so every variant in the notebooks' queues can be checked too. Default: the four main settings.

The other offline tests only run models for a handful of steps, so they cannot tell a model that trains from one
that does not. Here every job trains briefly on 256 training curves, and two things must improve:
  1. the model's own loss on those curves (not for the GAN, whose "loss" is a critic score);
  2. the values of generated curves must move towards those of the real curves: the 1-D Wasserstein distance
     between all generated and all real values (in amplitudes), with each family's sampling settings.
It is a sanity check of the training step and the sampler, not a measure of quality.
"""
import importlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astra import bench_data, bench_run  # noqa: E402
from astra.bench_eval import EVAL  # noqa: E402
from astra.bench_run import COMMON  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(4)
npz = sys.argv[1]
names = sys.argv[2:] or ["gan", "vae", "ddpm", "fm"]
STEPS = {"gan": 150, "vae": 400, "ddpm": 400, "fm": 400}
SAMPLING = {"gan": [{}], "vae": [{}, {"noise": False}], "ddpm": [{"nfe": 32}, {"nfe": 4}], "fm": [{"nfe": 32}, {"nfe": 4}]}
DATA = {}                                                        # one copy of the data per target ("unit", "raw")

failed = []
for name in names:
    family = name.split("_")[0]
    fam = importlib.import_module(f"astra.bench_{family}")
    job = name if "_" in name else f"{family}_{next(iter(fam.GRID))}_s0"
    cfg = dict(COMMON, **EVAL, **fam.DEFAULTS, kl_warmup_its=0)
    cfg.update(fam.GRID[bench_run.job_tag(job, family)])
    cfg.update(seed=0, ema=0.95, batch=64)                         # an EMA short enough for a short run
    if cfg["rep"] not in DATA:
        DATA[cfg["rep"]] = bench_data.load(npz, 0, "cpu", cfg["rep"])
    data = DATA[cfg["rep"]]
    sub = bench_data.balanced_idx(data, 256, seed=0)             # 256 training rows, every class equally often
    amp = data["amp"][sub][:, None]
    real = (data["x"][sub] / amp).flatten().sort().values

    def w1(model, **opts):
        """1-D Wasserstein distance between the generated and the real values (both in amplitudes)."""
        return float(((model.sample(sub, seed=2, **opts) / amp).flatten().sort().values - real).abs().mean())

    torch.manual_seed(0)
    model = fam.MODEL(cfg, data, "cpu")
    g = torch.Generator().manual_seed(0)
    draw = lambda: sub[torch.randint(len(sub), (cfg["batch"],), generator=g)]
    before = {"loss": model.losses(sub, seed=1), **{str(o): w1(model, **o) for o in SAMPLING[family]}}
    t0, n = time.time(), STEPS[family]
    for it in range(n):
        stats = model.step(draw, it)
    after = {"loss": model.losses(sub, seed=1), **{str(o): w1(model, **o) for o in SAMPLING[family]}}
    line = " | ".join(f"{k}: {before[k]:.3f} -> {after[k]:.3f}" for k in before)
    print(f"{job}: {n} steps in {time.time() - t0:.0f} s | last step {[round(v, 3) for v in stats.tolist()]} | {line}", flush=True)
    ok = all(after[str(o)] < 0.7 * before[str(o)] for o in SAMPLING[family] if o != {"noise": False})
    if family != "gan":
        ok = ok and after["loss"] < 0.8 * before["loss"]
    if not ok:
        failed.append(job)
assert not failed, f"no clear learning for: {failed}"
print("LEARNING TEST PASSED for:", ", ".join(names))
