# ---- Self-tests: harness ----
import importlib
from astra import bench_eval, bench_report, bench_run
from astra.sampling import ClassResampler
fam = importlib.import_module(f"astra.bench_{FAMILY}")

y = np.array([0] * 60 + [1] * 40)
pred = np.stack([y.copy(), np.zeros(100, int)])                  # run 0 is perfect, run 1 always says class 0
s = bench_eval.summarise(y, pred, 2, rare=np.array([False, True]))
assert np.allclose(s["bal_acc_runs"], [1.0, 0.5]) and np.allclose(s["acc_runs"], [1.0, 0.6])
b = bench_report.bootstrap_bal_acc(y, np.arange(100), {"good": pred[:1], "bad": pred[1:]}, 2, n_boot=200)
assert np.allclose(b["good"], 1.0) and np.allclose(b["bad"], 0.5)
assert all(j.startswith("real_s") or bench_run.job_tag(j, FAMILY) in fam.GRID for j in JOBS_ALL), "unknown job"

# the resume path on THIS device: a saved random-number state, loaded the way a resumed job loads it, must
# give the same random numbers again (on a GPU the loaded state has to be moved back to the CPU first)
from astra.common import restore_rng, torch_save
torch_save({"rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if DEV == "cuda" else None}, WORK / "rng_test.pt")
first = (torch.rand(3), torch.rand(3, device=DEV))
assert restore_rng(torch.load(WORK / "rng_test.pt", map_location=DEV, weights_only=False), DEV)
again = (torch.rand(3), torch.rand(3, device=DEV))
assert all(torch.equal(a, b) for a, b in zip(first, again)), "a resumed job would not continue its random numbers"
(WORK / "rng_test.pt").unlink()

# ---- this notebook's model (CPU, real-size networks, a batch of 16) ----
data_cpu = bench_data.load(NPZ, SEEDS[0], "cpu")
cfg = bench_run.job_config(SEARCH_JOBS[0], fam, "search", tiny=True)   # the same config path as the workers
torch.manual_seed(0)
model = fam.MODEL(cfg, data_cpu, "cpu")
rs = ClassResampler(data_cpu["y_np"][data_cpu["train_idx_np"]], cfg["gamma"], seed=0, n_classes=len(CLASSES))
stats = model.step(lambda: data_cpu["train_idx"][torch.from_numpy(rs.sample(16))], 0.5)
assert torch.isfinite(stats).all()
idx = bench_data.balanced_idx(data_cpu, 16, seed=0)
opts = {"nfe": 4} if "nfe" in cfg else {}
x1, x2 = model.sample(idx, seed=3, **opts), model.sample(idx, seed=3, **opts)
assert x1.shape == (16, L) and torch.equal(x1, x2), "sampling must repeat exactly for a fixed seed"
N_PARAMS = model.n_params()
print(f"harness tests passed | {FAMILY}: parameters {N_PARAMS} | one step: "
      + ", ".join(f"{n} {v:.3f}" for n, v in zip(model.STATS, stats.tolist())))
