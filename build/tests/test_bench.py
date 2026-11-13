"""Offline tests of the benchmark harness and trainer (CPU). Run from the project root:
    python build/tests/test_bench.py <catalina_L100.npz or ztf_L40.npz> [gan vae ddpm fm]
Outputs go to $ASTRA_WORK/bench_test.

1. Scores, bootstrap and split on cases with a known answer; the new conditions (other stars, same class).
2. Per family: a run stopped by the session time guard and resumed must end exactly where an uninterrupted
   run ends (same EMA weights, same checkpoint scores, same test scores, same new-condition scores).
3. Per family, on the tiny schedules: a search run stops by its rule and keeps its state; a final run in the
   search run's folder is refused; a final run in its own folder starts fresh on the final schedule and gets
   the test scores, the budget scores and the new-condition scores; its own-condition score at the selected
   checkpoint equals the sampling sweep's default part.
4. The result files feed every table and figure of the report.
"""
import importlib
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astra import bench_data, bench_eval, bench_report, bench_run  # noqa: E402
from astra.common import WORK, Deadline, load_json, save_json  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(4)
npz = sys.argv[1]
families = sys.argv[2:] or ["gan", "vae", "ddpm", "fm"]
root = WORK / "bench_test"
shutil.rmtree(root, ignore_errors=True)

# ---- 1. scores, bootstrap, split ----
y = np.array([0] * 60 + [1] * 40)
pred = np.stack([y.copy(), np.zeros(100, int)])                       # run 0: perfect; run 1: always class 0
s = bench_eval.summarise(y, pred, 2, rare=np.array([False, True]))
assert np.allclose(s["bal_acc_runs"], [1.0, 0.5]) and np.allclose(s["acc_runs"], [1.0, 0.6])
assert np.isclose(s["rare_recall"], 0.5) and np.isclose(s["common_recall"], 1.0)
b = bench_report.bootstrap_bal_acc(y, np.arange(100), {"good": pred[:1], "bad": pred[1:]}, 2, n_boot=200)
assert np.allclose(b["good"], 1.0) and np.allclose(b["bad"], 0.5), "balanced accuracy is 0.5 for a constant guess"
data = bench_data.load(npz, 0, "cpu")
sides = [set(data["star"][data[f"{s_}_idx_np"]]) for s_ in ("train", "val", "test")]
assert not (sides[0] & sides[1]) and not (sides[0] & sides[2]) and not (sides[1] & sides[2]), "a star is on two sides"
share = [len(data[f"{s_}_idx_np"]) / len(data["y_np"]) for s_ in ("train", "val", "test")]
assert abs(share[0] - 0.7) < 0.02 and abs(share[1] - 0.1) < 0.02 and abs(share[2] - 0.2) < 0.02, share
assert bench_data.load(npz, 0, "cpu")["hash"] == data["hash"] != bench_data.load(npz, 1, "cpu")["hash"]
assert float(data["u"].abs().max()) <= 2.0 + 1e-5
# the ablation switch: with use_phase=False the network's output must not depend on the phases, with True it must
from astra.bench_net import UNet1d  # noqa: E402
torch.manual_seed(0)
xq, tq, cq, yq = torch.randn(4, data["L"]), torch.rand(4), torch.randn(4), torch.zeros(4, dtype=torch.long)
pa, pb = data["ph"][:4], data["ph"][4:8]
for use, want in ((True, True), (False, False)):
    net = UNet1d(data["n_classes"], 16, use_phase=use)
    torch.nn.init.normal_(net.out.weight)
    for b in list(net.enc0) + list(net.dec0):
        torch.nn.init.normal_(b.c2.weight, std=0.1)
    with torch.no_grad():
        differs = not torch.allclose(net(xq, tq, pa, cq, yq), net(xq, tq, pb, cq, yq), atol=1e-6)
    assert differs == want, f"use_phase={use}: output depends on phases = {differs}"
# new conditions: the phases of another training star of the same class, the amplitude of a third one
idx_n = bench_data.balanced_idx(data, 600, seed=3)
view, ph_from, amp_from = bench_eval.novel_conditions(data, idx_n, seed=0)
rows_n, st, yy = idx_n.numpy(), data["star"], data["y_np"]
assert (yy[ph_from] == yy[rows_n]).all() and (yy[amp_from] == yy[rows_n]).all(), "a donor of another class"
assert (st[ph_from] != st[rows_n]).all() and (st[amp_from] != st[rows_n]).all() and (st[amp_from] != st[ph_from]).all()
assert set(ph_from) | set(amp_from) <= set(data["train_idx_np"]), "a donor outside the training stars"
assert torch.equal(view["ph"], data["ph"][ph_from]) and torch.equal(view["mag_scale"], data["mag_scale"][amp_from])
assert torch.equal(view["y"], data["y"][idx_n]) and "x" not in view and "u" not in view
assert np.array_equal(bench_eval.novel_conditions(data, idx_n, seed=0)[1], ph_from), "not reproducible"
print(f"1 scores, bootstrap, split ok | train/val/test {[len(data[f'{s_}_idx_np']) for s_ in ('train', 'val', 'test')]} "
      f"| training curves per class {data['train_counts'].tolist()} | rare tier {np.array(data['classes'])[data['rare']].tolist()}")

# ---- 2. interrupted + resumed == uninterrupted ----
SMALL = dict(iters=12, n_ckpt=3, chunk=2, n_sel=150, n_mem=60, n_loss=60, n_final=200, max_sweeps=0, p_syn=(0.5,),
             sel_max_epochs=0.25, final_max_epochs=0.25)


class StopAt(Deadline):
    """Pretends the session time is up once the run reaches iteration `n` (only the first time)."""

    def __init__(self, ref, n):
        super().__init__(1e12)
        self.ref, self.n, self.fired = ref, n, False

    def passed(self, reserve_s=0.0):
        if not self.fired and self.ref[0] is not None and self.ref[0].it >= self.n:
            self.fired = True
            return True
        return False


feat = bench_run.feature_cnn(root, data, "cpu", max_epochs=1)
fid = bench_eval.Fidelity(data, feat)
for name in families:
    fam = importlib.import_module(f"astra.bench_{name}")
    job = f"{name}_{next(iter(fam.GRID))}_s0"
    cfg = bench_run.job_config(job, fam, "final", tiny=True)
    cfg.update(SMALL)
    n = cfg["n_ckpt"]
    cfg.update(ckpt_its=[round(cfg["iters"] * (k + 1) / n) for k in range(n)], cap_its=cfg["iters"], patience=None)
    ref = [None]
    r1 = bench_run.BenchRun(cfg, data, fid, fam.MODEL, root / "interrupted" / job, StopAt(ref, 6), "cpu")
    ref[0] = r1
    assert r1.run(reserve_s=0) == "paused" and r1.it == 6 and r1.n_ckpt == 1, (r1.it, r1.n_ckpt)
    r2 = bench_run.BenchRun(cfg, data, fid, fam.MODEL, root / "interrupted" / job, Deadline(1e12), "cpu")
    assert r2.run(reserve_s=0) == "done" and r2.it == 12
    r3 = bench_run.BenchRun(cfg, data, fid, fam.MODEL, root / "runs" / job, Deadline(1e12), "cpu")
    assert r3.run(reserve_s=0) == "done"
    k = load_json(root / "runs" / job / "final.json")["selected_ckpt"]
    for d in ("interrupted", "runs"):
        assert sorted(p.name for p in (root / d / job / "ckpt").glob("*.pt")) == ["ema_01.pt", "ema_02.pt", "ema_03.pt"]
    a, b_ = (torch.load(root / d / job / "ckpt" / "ema_03.pt") for d in ("interrupted", "runs"))
    for net in a:
        for u, v in zip(a[net].values(), b_[net].values()):
            assert torch.equal(u, v), f"{name}: the resumed run differs from the uninterrupted run"
    m2, m3 = load_json(root / "interrupted" / job / "metrics.json"), load_json(root / "runs" / job / "metrics.json")
    assert [m["ckpt"] for m in m2] == [1, 2, 3]
    assert [m["val_bal_acc"] for m in m2] == [m["val_bal_acc"] for m in m3] and [m["fid"] for m in m2] == [m["fid"] for m in m3]
    f2, f3 = load_json(root / "interrupted" / job / "final.json"), load_json(root / "runs" / job / "final.json")
    assert f2["parts"]["tstr"]["bal_acc_runs"] == f3["parts"]["tstr"]["bal_acc_runs"] and f2["selected_ckpt"] == k
    assert f2["parts"]["novel_selected"]["bal_acc_runs"] == f3["parts"]["novel_selected"]["bal_acc_runs"]
    assert not (root / "runs" / job / "last.pt").exists()
    bench_run.write_result(job, name, "test", cfg, data, root / "runs" / job, root / "results_test")
    print(f"2 {name}: resume ok | selected checkpoint {k} | parameters {r3.model.n_params()} | "
          f"TSTR balanced accuracy {f3['parts']['tstr']['bal_acc']:.3f} | parts {sorted(f3['parts'])}")

# ---- 3. search stage, stopping rule, a fresh final stage in its own folder, budget and new-condition scores ----
for name in families:
    fam = importlib.import_module(f"astra.bench_{name}")
    tag = next(iter(fam.GRID))
    job = f"{name}_{tag}_s0"
    small = {k: v for k, v in SMALL.items() if k not in ("iters", "n_ckpt")}
    cs = dict(bench_run.job_config(job, fam, "search", tiny=True), **small)
    cf = dict(bench_run.job_config(job, fam, "final", tiny=True), **small)
    assert cs["ckpt_its"] == [12, 24, 36] and cf["ckpt_its"] == [6, 12, 24, 36, 48], (cs["ckpt_its"], cf["ckpt_its"])
    out, out_f = root / "stages" / job, root / "stages_final" / job
    rs = bench_run.BenchRun(cs, data, fid, fam.MODEL, out, Deadline(1e12), "cpu")
    assert rs.run(reserve_s=0) == "done" and (out / "search.json").exists() and (out / "last.pt").exists()
    sj = load_json(out / "search.json")
    m = load_json(out / "metrics.json")
    best = max(range(len(m)), key=lambda i: m[i]["val_bal_acc"])
    # the rule: stop at the cap, or once `patience` (1 here) checkpoints after the best bring no gain
    assert sj["stop"] in ("cap", "no gain") and (sj["stop"] == "cap") == (rs.it == 36), (sj, rs.it)
    assert sj["stop"] == "cap" or len(m) - 1 - best >= 1, (sj, [a["val_bal_acc"] for a in m])
    assert all("copy_share" in a and "solo_h" in a and "it" in a for a in m)
    bench_run.write_result(job, name, "test", cs, data, out, root / "results_stages")
    try:            # the final stage must never continue a search run (it did until 2026-10-22)
        bench_run.BenchRun(cf, data, fid, fam.MODEL, out, Deadline(1e12), "cpu").run(reserve_s=0)
        raise AssertionError(f"{name}: a final run in the search run's folder was not refused")
    except RuntimeError:
        pass
    rf = bench_run.BenchRun(cf, data, fid, fam.MODEL, out_f, Deadline(1e12), "cpu")
    assert rf.run(reserve_s=0) == "done"
    log_f = (out_f / "log.txt").read_text(encoding="utf-8")
    mf = load_json(out_f / "metrics.json")
    assert "resumed" not in log_f and "parameters:" in log_f and mf[0]["it"] == 6, "the final run did not start fresh"
    assert [a["it"] for a in mf] == cf["ckpt_its"][:len(mf)]
    fj = load_json(out_f / "final.json")
    assert fj["stage"] == "final" and fj["selected_ckpt"] == max(mf, key=lambda a: a["val_bal_acc"])["ckpt"]
    assert set(fj["budgets"]) == {"0.3min"} and fj["budgets"]["0.3min"]["it"] <= 18, fj["budgets"]
    bt = fj["budgets"]["0.3min"]["test"]
    assert bt == "tstr" or bt in fj["parts"], fj["budgets"]
    P = fj["parts"]
    assert P["novel_selected"]["novel"] and P["novel_selected"]["ckpt"] == fj["selected_ckpt"]
    assert fj["last_ckpt"] == mf[-1]["ckpt"]
    assert (fj["last_ckpt"] == fj["selected_ckpt"]) == ("novel_last" not in P), (fj["last_ckpt"], fj["selected_ckpt"], sorted(P))
    assert "novel_last" not in P or (P["novel_last"]["novel"] and not P["default_last"]["novel"]
                                      and P["novel_last"]["ckpt"] == P["default_last"]["ckpt"] == fj["last_ckpt"])
    assert not (out_f / "last.pt").exists()
    # the own-condition score at the selected checkpoint is the sampling sweep's default part, number for number
    rf.model.load_sampler_state(torch.load(out_f / "ckpt" / f"ema_{fj['selected_ckpt']:02d}.pt"))
    own = bench_eval.condition_test(rf.model, data, cf, out_f, Deadline(1e12), "cpu", lambda *a: None, "own_check",
                                    novel=False, ckpt=fj["selected_ckpt"])
    assert own["bal_acc_runs"] == P["sweep_default"]["bal_acc_runs"], (own["bal_acc_runs"], P["sweep_default"]["bal_acc_runs"])
    bench_run.write_result(job, name, "test", cf, data, out_f, root / "results_stages")
    print(f"3 {name}: search stops by '{sj['stop']}' at {rs.it} steps; final in its own folder, fresh, to {rf.it}; "
          f"budget checkpoint {fj['budgets']['0.3min']['ckpt']} ({bt}); new conditions at checkpoints "
          f"{fj['selected_ckpt']} and {fj['last_ckpt']}: {sorted(k for k in P if 'novel' in k or 'last' in k)}")
srch = bench_report.collect([root], "results_stages", stage="search")
fin = bench_report.collect([root], "results_stages")
assert set(fin) == {f"{f}_{next(iter(importlib.import_module(f'astra.bench_{f}').GRID))}_s0" for f in families}
print(f"--- search table ---\n{bench_report.search_table(srch).to_string(index=False)}" if srch else "(search results "
      "replaced by the final ones, as on Kaggle)")
print(f"--- budget table ---\n{bench_report.budget_table(fin).to_string(index=False)}")
print(f"--- new conditions ---\n{bench_report.novel_table(fin).to_string(index=False)}")
assert len(bench_report.budget_table(fin)) and len(bench_report.novel_table(fin))
bench_report.plot_budget(fin, root / "budgets.png")

# ---- 4. the reference rows and the report ----
cfg = dict(bench_run.job_config("real_s0", None, "final", tiny=True), **SMALL)
log = lambda *a: print(" ", *a)
assert bench_eval.run_real(data, cfg, root / "runs" / "real_s0", Deadline(1e12), "cpu", log) is not None
save_json(cfg, root / "runs" / "real_s0" / "config.json")
bench_run.write_result("real_s0", "real", "test", cfg, data, root / "runs" / "real_s0", root / "results_test")
res = bench_report.collect([root], "results_test")
assert set(res) == {"real_s0"} | {f"{f}_{next(iter(importlib.import_module(f'astra.bench_{f}').GRID))}_s0" for f in families}
for title, t in (("jobs", bench_report.job_table(res, n_boot=200)), ("real rows", bench_report.real_table(res)),
                 ("families", bench_report.family_table(res)), ("classes", bench_report.class_table(res)),
                 ("pairs", bench_report.family_pairs(res, n_boot=200)), ("sweeps", bench_report.sweep_table(res)),
                 ("new conditions", bench_report.novel_table(res))):
    print(f"\n--- {title} ---\n{t.to_string(index=title == 'classes')}")
    assert len(t) or (title == "pairs" and len(families) < 2)
assert all("[" in v for v in bench_report.job_table(res, n_boot=50)["TSTR - A [95% CI]"]), "paired bootstrap missing"
Z = np.load(npz)
bench_report.plot_training(res, root / "training.png")
bench_report.plot_classifiers(res, root / "classifiers.png")
bench_report.plot_sweeps(res, root / "sweeps.png")
bench_report.plot_samples(root / "runs" / next(j for j in res if j != "real_s0"), Z, "samples", root / "samples.png")
assert all((root / f).exists() for f in ("training.png", "classifiers.png", "sweeps.png", "samples.png"))
print("\nBENCH TESTS PASSED (interrupted + resumed == uninterrupted, bit for bit, for:", ", ".join(families) + ")")
