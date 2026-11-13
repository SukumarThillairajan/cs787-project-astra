"""Trainer and per-GPU worker of the ASTRA benchmark, shared by the four model families.

python -m astra.bench_run --family fm --stage search --queue fm_un_lr2e-4_s0 --deadline <unix time> --data <npz>
                          --dataset catalina --runs-dir <dir> --results-dir <dir> [--smoke]
python -m astra.bench_run --family fm --speedtest 300 --data <npz>        (SMOKE: training steps per second)

The controls of the comparison live here, so they are identical for GAN, VAE, DDPM and flow matching:
  - the same schedule: checkpoints after 5, 10, 20, 30, 45, 60, 90, ... minutes of training ON ONE T4 ALONE
    (search), after 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, ... minutes (final: the diffusion-type models
    peak within 10 minutes, before they start copying their training curves). The minutes are turned into
    iterations with the configuration's measured speed (SPEED in astra/bench_grid.py), so the schedule does not
    change when two jobs share a GPU and a resumed run repeats exactly;
  - the same stopping rule: a run trains until 4 checkpoints in a row bring no new best validation score (and at
    least 1 h), or until the cap of its stage (search 6 h, final 20 h);
  - the same batches: the paper's resampling block with gamma = 0.25; every checkpoint is the EMA (0.999);
  - the same model selection: the checkpoint with the best validation TSTR balanced accuracy;
  - the same compute-matched view: the best checkpoint within 5, 10, 15, 30, 60 min and 2, 5, 10 h is also
    scored on test;
  - the same final evaluation (astra.bench_eval), including TSTR for conditions no training star has, at the
    selected and at the last checkpoint.
Two stages. "search": every configuration of the family's GRID on split 0, validation scores only. "final": the
winning configuration on splits 0-4 with test scores; every final run starts fresh (the notebooks keep the final
runs in their own folder, so split 0 does not continue the search run; since 2026-10-22). A family module
supplies DEFAULTS, GRID, FINAL, SPEED, JOBS, TINY and a Family subclass named MODEL.
"""
import argparse
import copy
import hashlib
import importlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import bench_data
from .bench_eval import (EVAL, EVAL_TINY, Fidelity, condition_test, evaluate_checkpoint, final_evaluation, run_real,
                         save_npz, tstr_only)
from .classifier import CNNEnsemble, amp_on, train_feature_cnn
from .common import WORK, Deadline, Logger, load_json, restore_rng, save_json, torch_save
from .metrics import g_score
from .sampling import ClassResampler

STAGES = ("search", "final")
LATER_MIN = (60, 90, 120, 180, 240, 300, 360, 420, 480, 540, 600, 720, 840, 960, 1080, 1200)
COMMON = dict(gamma=0.25, ema=0.999, amp=True, rep="unit",
              # checkpoints of each stage, in minutes of training on one T4 alone (turned into iterations with
              # `speed`); the final stage looks closer at the first 10 minutes (decided 2026-10-21)
              ckpt_min={"search": (5, 10, 20, 30, 45) + LATER_MIN,
                        "final": (1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45) + LATER_MIN},
              cap_h={"search": 6.0, "final": 20.0},     # training cap of each stage, same units
              patience=4, min_h=1.0,          # stop when 4 checkpoints in a row bring no new best (after >= 1 h)
              # compute-matched view: the best checkpoint within each budget, in minutes on one T4 alone
              budgets_min=(5, 10, 15, 30, 60, 120, 300, 600),
              speed=None,                     # steps per second alone on a T4 (from the family's SPEED table)
              iters=None, n_ckpt=None,        # tests only: a fixed number of steps with n_ckpt equal checkpoints
              chunk=20, save_every_s=1800)
# smoke and offline tests: search checkpoints at 12, 24, 36 steps, final at 6, 12, 24, 36, 48
TINY = dict(ckpt_min={"search": (0.2, 0.4, 0.6, 0.8), "final": (0.1, 0.2, 0.4, 0.6, 0.8)},
            cap_h={"search": 0.6 / 60, "final": 0.8 / 60}, patience=1, min_h=0.0, budgets_min=(0.3,), speed=1.0,
            save_every_s=20)


def grid_jobs(prefix, grid, final):
    """Job names of a family: every grid configuration on split 0 (the search), and the final configuration
    on splits 1-4 (its split 0 has the search run's name; the final stage runs it fresh in its own folder)."""
    jobs = {f"{prefix}_{tag}_s0": dict(seed=0, **cfg) for tag, cfg in grid.items()}
    if final:
        jobs.update({f"{prefix}_{final}_s{s}": dict(seed=s, **grid[final]) for s in (1, 2, 3, 4)})
    return jobs


def budget_key(minutes):
    """The name of a compute budget in the results: 30 -> '30min' (astra.bench_report.budget_hours reads it)."""
    return f"{minutes:g}min"


def job_tag(job, family):
    """gan_lr1e-4_b64_s0 -> lr1e-4_b64"""
    return job[len(family) + 1:job.rindex("_s")]


class Family:
    """What a model family provides to the trainer. A subclass builds its networks in __init__, registers
    them with track() and make_ema(), and implements step(), sample_batch(), losses() and eval_nets()."""
    SWEEPS = ()                        # other sampling settings to score at the selected checkpoint
    STATS = ("loss",)                  # names of the numbers step() returns

    def __init__(self, cfg, data, device):
        self.cfg, self.data, self.dev = cfg, data, device
        self.amp = amp_on(cfg.get("amp", True), device)
        self._tracked, self._ema, self.n_bad = {}, [], 0

    def track(self, **objs):
        """Networks, optimisers and gradient scalers: everything a resumed run must restore."""
        self._tracked.update(objs)

    def make_ema(self, net):
        ema = copy.deepcopy(net).eval().requires_grad_(False)
        self._ema.append((ema, net))
        return ema

    def update_ema(self):
        with torch.no_grad():
            for ema, net in self._ema:
                for pe, p in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(p, 1 - self.cfg["ema"])

    def state_dict(self):
        return {k: v.state_dict() for k, v in self._tracked.items()}

    def load_state_dict(self, s):
        for k, v in self._tracked.items():
            v.load_state_dict(s[k])

    def eval_nets(self):
        """name -> network, for everything sample() and losses() use (this is what a checkpoint stores)."""
        raise NotImplementedError

    def sampler_state(self):
        return {k: copy.deepcopy(v.state_dict()) for k, v in self.eval_nets().items()}

    def load_sampler_state(self, s):
        for k, v in self.eval_nets().items():
            v.load_state_dict(s[k])

    def n_params(self):
        return {k: sum(p.numel() for p in v.parameters()) for k, v in self._tracked.items()
                if isinstance(v, torch.nn.Module)}

    def cond(self, idx):
        d = self.data
        return d["ph"][idx], d["camp"][idx], d["y"][idx]

    def to_mag(self, u, idx):
        return u.float() * self.data["mag_scale"][idx][:, None]

    @torch.no_grad()
    def sample(self, idx, seed, batch=4096, **opts):
        """Magnitudes of one generated curve per condition row idx[i] (EMA weights, full precision output)."""
        g = torch.Generator(device=self.dev).manual_seed(seed)
        x = torch.cat([self.sample_batch(idx[i:i + batch], g, **opts) for i in range(0, len(idx), batch)])
        bad = ~torch.isfinite(x)
        self.n_bad = int(bad.sum())        # reported with the scores; a non-finite value is replaced by 0
        return torch.where(bad, torch.zeros_like(x), x) if self.n_bad else x


class preserve_rng:
    """Evaluation must not change the random numbers of training (a resumed run then repeats exactly)."""

    def __enter__(self):
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def __exit__(self, *exc):
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)


class BenchRun:
    """One training run of one family: schedule, checkpoints, their evaluation, stopping rule, resume, scores."""

    def __init__(self, cfg, data, fidelity, family_cls, out, deadline, device):
        self.cfg, self.data, self.fid, self.dev = cfg, data, fidelity, device
        self.out, self.deadline = out, deadline
        (out / "ckpt").mkdir(parents=True, exist_ok=True)
        self.log = Logger(out / "log.txt")
        self.metrics = load_json(out / "metrics.json", [])
        self.ckpts = load_json(out / "ckpts.json", [])           # iteration and training seconds of each checkpoint
        torch.manual_seed(cfg["seed"])
        self.model = family_cls(cfg, data, device)
        self.sampler = ClassResampler(data["y_np"][data["train_idx_np"]], cfg["gamma"], seed=cfg["seed"],
                                      n_classes=data["n_classes"])
        self.it, self.train_s, self.n_ckpt = 0, 0.0, 0
        self.hist, self.n_hist = None, 0
        self.sched, self.cap = cfg["ckpt_its"], cfg["cap_its"]

    def draw(self):
        """Row indices of one training batch from the resampling block."""
        return self.data["train_idx"][torch.from_numpy(self.sampler.sample(self.cfg["batch"])).to(self.dev)]

    def sync(self):
        if str(self.dev).startswith("cuda"):
            torch.cuda.synchronize()

    def solo_h(self, it):
        return it / self.cfg["speed"] / 3600 if self.cfg.get("speed") else float("nan")

    # ---------------- state ----------------
    def save_state(self):
        torch_save({"model": self.model.state_dict(), "sampler": self.sampler, "it": self.it,
                    "train_s": self.train_s, "n_ckpt": self.n_ckpt, "hist": self.hist, "n_hist": self.n_hist,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if str(self.dev).startswith("cuda") else None},
                   self.out / "last.pt")

    def load_state(self):
        p = self.out / "last.pt"
        if not p.exists():
            return False
        s = torch.load(p, map_location=self.dev, weights_only=False)
        self.model.load_state_dict(s["model"])
        self.sampler, self.it, self.train_s, self.n_ckpt = s["sampler"], s["it"], s["train_s"], s["n_ckpt"]
        self.hist, self.n_hist = s["hist"], s["n_hist"]
        restore_rng(s, self.dev)
        return True

    # ---------------- training ----------------
    def next_ckpt(self):
        return self.sched[self.n_ckpt] if self.n_ckpt < len(self.sched) else self.cap

    def train_chunk(self):
        n = min(self.cfg["chunk"], self.next_ckpt() - self.it)     # land exactly on the checkpoint's step
        self.sync()
        t0 = time.time()
        for _ in range(n):
            s = self.model.step(self.draw, self.it)
            self.hist = s if self.hist is None else self.hist + s
            self.n_hist += 1
            self.it += 1
        self.sync()
        self.train_s += time.time() - t0   # only training counts; evaluation does not

    def checkpoint(self):
        self.n_ckpt += 1
        torch_save(self.model.sampler_state(), self.out / "ckpt" / f"ema_{self.n_ckpt:02d}.pt")
        self.ckpts = [c for c in self.ckpts if c["ckpt"] < self.n_ckpt] + [
            {"ckpt": self.n_ckpt, "it": self.it, "train_s": self.train_s, "solo_h": self.solo_h(self.it)}]
        save_json(self.ckpts, self.out / "ckpts.json")
        h = (self.hist / max(self.n_hist, 1)).tolist()
        stats = ", ".join(f"{n} {v:.4f}" for n, v in zip(self.model.STATS, h))
        self.log(f"checkpoint {self.n_ckpt}: iteration {self.it:,} = {self.solo_h(self.it):.2f} h on one T4 alone "
                 f"({self.train_s / 60:.1f} min here, {self.it / max(self.train_s, 1e-9):.1f} it/s) | {stats}")
        with open(self.out / "train_curve.csv", "a") as f:
            f.write(f"{self.n_ckpt},{self.it},{self.train_s:.1f}," + ",".join(f"{v:.6f}" for v in h) + "\n")
        self.hist, self.n_hist = None, 0
        self.save_state()

    def evaluate_pending(self):
        done = {m["ckpt"] for m in self.metrics}
        todo = [k for k in range(1, self.n_ckpt + 1) if k not in done]
        if not todo:
            return
        live = self.model.sampler_state()
        meta = {c["ckpt"]: c for c in self.ckpts}
        with preserve_rng():
            for k in todo:
                self.model.load_sampler_state(torch.load(self.out / "ckpt" / f"ema_{k:02d}.pt", map_location=self.dev))
                m = evaluate_checkpoint(self.model, self.fid, self.data, self.cfg, k, self.dev)
                m.update({key: meta[k][key] for key in ("it", "train_s", "solo_h")})
                self.metrics.append(m)
                self.metrics.sort(key=lambda a: a["ckpt"])
                save_json(self.metrics, self.out / "metrics.json")
                self.log(f"checkpoint {k}: val TSTR balanced acc {100 * m['val_bal_acc']:.2f} | FID {m['fid']:.2f} | "
                         f"density {np.nanmean(m['density']):.3f} recall {np.nanmean(m['recall']):.3f} | "
                         f"memorisation ratio {np.nanmean(m['mem_ratio']):.2f}, copies "
                         f"{100 * np.nanmean(m['copy_share']):.1f}% | loss train {m['loss_train']:.4f} "
                         f"val {m['loss_val']:.4f} ({m['eval_s']:.0f} s)"
                         + (f" | !! {m['nonfinite']} non-finite generated values set to 0" if m["nonfinite"] else ""))
        self.model.load_sampler_state(live)

    def stop_reason(self):
        """None while the run goes on; 'cap' or 'no gain' when it ends (the metrics are up to date here)."""
        c, m = self.cfg, self.metrics
        if c.get("patience") and len(m) > c["patience"] and self.it >= c["min_its"]:
            best = int(np.argmax([a["val_bal_acc"] for a in m]))
            if len(m) - 1 - best >= c["patience"]:
                return "no gain"          # checked first: a run that stops at its cap may continue in the next stage
        return "cap" if self.it >= self.cap else None

    # ---------------- the whole run ----------------
    def run(self, reserve_s):
        c, stage = self.cfg, self.cfg["stage"]
        if (self.out / "final.json").exists() or (stage == "search" and (self.out / "search.json").exists()):
            return "done"
        if stage == "final" and (self.out / "search.json").exists():
            raise RuntimeError(f"{self.out} holds a search run: final runs start fresh in the final stage's own folder")
        if self.load_state():
            self.log(f"{stage} stage resumed at iteration {self.it:,} ({self.solo_h(self.it):.2f} h on one T4 alone)")
        else:            # a fresh run: its random numbers must not depend on what ran between __init__ and here
            torch.manual_seed(c["seed"] + 10_000)
            self.log(f"parameters: {self.model.n_params()} | schedule (iterations): {self.sched} | cap {self.cap:,}")
        self.evaluate_pending()
        t_save = time.time()
        while True:
            why = self.stop_reason()
            if why:
                break
            if self.deadline.passed(reserve_s):
                self.save_state()
                self.log(f"time budget of the session reached at iteration {self.it:,}: state saved, resume next session")
                return "paused"
            self.train_chunk()
            if self.it >= self.next_ckpt():
                self.checkpoint()
                self.evaluate_pending()
                t_save = time.time()
            elif time.time() - t_save > c["save_every_s"]:
                self.save_state()
                t_save = time.time()
        return self.finish_search(why) if stage == "search" else self.finish(why)

    def best(self, max_it=None):
        m = [a for a in self.metrics if max_it is None or a["it"] <= max_it]
        return max(m, key=lambda a: a["val_bal_acc"]) if m else None

    def finish_search(self, why):
        """The search scores validation only. The state is kept (the final stage starts its own runs)."""
        b = self.best()
        self.save_state()
        save_json({"stage": "search", "stop": why, "iters": self.it, "train_s": self.train_s,
                   "solo_h": self.solo_h(self.it), "best_ckpt": b["ckpt"], "best_val_bal_acc": b["val_bal_acc"],
                   "best_it": b["it"], "best_solo_h": b["solo_h"], "n_params": self.model.n_params()},
                  self.out / "search.json")
        self.log(f"search run ends ({why}) at {self.solo_h(self.it):.2f} h: best checkpoint {b['ckpt']} at "
                 f"{b['solo_h']:.2f} h, val TSTR balanced acc {100 * b['val_bal_acc']:.2f}")
        return "done"

    def finish(self, why):
        """Model selection on validation stars, then the test scores of the selected checkpoint, then the
        compute-matched view: the best checkpoint within each budget, scored on test (TSTR), then TSTR for
        conditions no training star has, at the selected and at the last checkpoint."""
        m = self.metrics
        gs = g_score(np.nan_to_num([a["density"] for a in m]), np.nan_to_num([a["recall"] for a in m]))
        for a, g in zip(m, gs):
            a["g_score"] = float(g)
        save_json(m, self.out / "metrics.json")
        best = self.best()
        self.model.load_sampler_state(torch.load(self.out / "ckpt" / f"ema_{best['ckpt']:02d}.pt", map_location=self.dev))
        self.log(f"run ends ({why}) at {self.solo_h(self.it):.2f} h; selected checkpoint {best['ckpt']} at "
                 f"{best['solo_h']:.2f} h (val TSTR balanced acc {100 * best['val_bal_acc']:.2f}); "
                 f"the G-score would pick checkpoint {m[int(np.argmax(gs))]['ckpt']}")
        with preserve_rng():
            parts = final_evaluation(self.model, self.fid, self.data, self.cfg, self.out, self.deadline, self.dev,
                                     self.log, sweeps=self.model.SWEEPS[:self.cfg["max_sweeps"]])
        if parts is None:
            return "paused"
        budgets = {}
        for B in self.cfg["budgets_min"]:
            b = self.best(max_it=round(B * 60 * self.cfg["speed"]))
            if b is None:
                continue
            key = budget_key(B)
            budgets[key] = {"ckpt": b["ckpt"], "it": b["it"], "solo_h": b["solo_h"], "val_bal_acc": b["val_bal_acc"],
                            "mem_ratio": b["mem_ratio"], "copy_share": b["copy_share"]}
            if b["ckpt"] == best["ckpt"]:
                budgets[key]["test"] = "tstr"                  # the same checkpoint as the headline
                continue
            self.model.load_sampler_state(torch.load(self.out / "ckpt" / f"ema_{b['ckpt']:02d}.pt", map_location=self.dev))
            with preserve_rng():
                got = tstr_only(self.model, self.data, self.cfg, self.out, self.deadline, self.dev, self.log,
                                name=f"budget_{key}")
            if got is None:
                return "paused"
            budgets[key]["test"] = f"budget_{key}"
        # does a model that copies still generate useful curves for conditions it has not seen? (n_sel curves,
        # 3 classifier runs, as the sampling sweeps; the selected checkpoint's own conditions = sweep_default)
        last = m[-1]["ckpt"]
        novel = [("novel_selected", best["ckpt"], True)]
        if last != best["ckpt"]:
            novel += [("default_last", last, False), ("novel_last", last, True)]
        for name, k, new in novel:
            self.model.load_sampler_state(torch.load(self.out / "ckpt" / f"ema_{k:02d}.pt", map_location=self.dev))
            with preserve_rng():
                got = condition_test(self.model, self.data, self.cfg, self.out, self.deadline, self.dev, self.log,
                                     name, novel=new, ckpt=k)
            if got is None:
                return "paused"
        parts = load_json(self.out / "final_partial.json", {})
        save_json({"stage": "final", "stop": why, "selected_ckpt": best["ckpt"], "selected_solo_h": best["solo_h"],
                   "last_ckpt": last, "last_solo_h": m[-1]["solo_h"],
                   "g_score_ckpt": m[int(np.argmax(gs))]["ckpt"], "iters": self.it, "train_s": self.train_s,
                   "solo_h": self.solo_h(self.it), "n_params": self.model.n_params(), "budgets": budgets,
                   "parts": parts}, self.out / "final.json")
        (self.out / "last.pt").unlink(missing_ok=True)
        self.log("finished")
        return "done"


def feature_cnn(runs_dir, data, device, max_epochs=20, log=print):
    """The classifier whose last conv block is the feature space of the fidelity scores: trained once per
    split on the real training curves (as in the GAN notebook), then reused by every job of that split."""
    path = runs_dir / f"feature_cnn_s{data['seed']}.pt"
    model = CNNEnsemble(1, data["n_classes"], data["L"]).to(device)
    if not path.exists():
        tr, va = (bench_data.subset(data, data[f"{s}_idx"]) for s in ("train", "val"))
        model, curve = train_feature_cnn(tr, va, data["n_classes"], max_epochs=max_epochs, seed=data["seed"],
                                         device=device, log=log)
        torch_save(model.state_dict(), path)
        save_json(curve, runs_dir / f"feature_cnn_s{data['seed']}_curve.json")
    model.load_state_dict(torch.load(path, map_location=device))
    h = hashlib.md5()
    for v in model.state_dict().values():
        h.update(v.detach().cpu().numpy().tobytes())
    data["feat_hash"] = h.hexdigest()[:12]      # stored with every result that uses this feature space
    return model.eval()


def job_config(job, fam, stage, tiny, speeds=None):
    """real_s<seed> = the reference rows on real data; any other name is looked up in the family's JOBS.
    The schedule in minutes becomes iterations here, with the configuration's measured speed."""
    assert stage in STAGES, stage
    cfg = dict(COMMON, **EVAL, seed=int(job.rsplit("_s", 1)[1]), job=job, stage=stage)
    if job.startswith("real_s"):
        return dict(cfg, **(dict(TINY, **EVAL_TINY, tiny=True) if tiny else {}))
    cfg.update(fam.DEFAULTS)
    cfg["tag"] = job_tag(job, fam.__name__.rsplit("_", 1)[1])
    cfg.update(fam.JOBS.get(job) or fam.GRID[cfg["tag"]])     # any grid configuration on any split (SMOKE, tests)
    if tiny:
        cfg.update(TINY, **EVAL_TINY, tiny=True)
        cfg.update(getattr(fam, "TINY", {}))
    else:
        cfg["speed"] = fam.SPEED.get(cfg["tag"]) or (speeds or {}).get(cfg["tag"])
        if not cfg["speed"]:
            raise ValueError(f"{job}: no measured speed for {cfg['tag']} (SPEED in astra/bench_grid.py, or "
                             f"runs_*/speeds.json of the search output)")
    spd = cfg["speed"]
    if cfg.get("iters"):                       # tests: a fixed number of steps, n_ckpt equal checkpoints, no early stop
        n = cfg["n_ckpt"]
        cfg["ckpt_its"] = [round(cfg["iters"] * (k + 1) / n) for k in range(n)]
        cfg["cap_its"], cfg["patience"] = cfg["iters"], None
    else:
        cap = round(cfg["cap_h"][stage] * 3600 * spd)
        its = sorted({max(1, round(m * 60 * spd)) for m in cfg["ckpt_min"][stage]})
        cfg["ckpt_its"] = [i for i in its if i < cap] + [cap]
        cfg["cap_its"] = cap
    cfg["min_its"] = round(cfg["min_h"] * 3600 * spd)
    cfg["kl_warmup_its"] = round(cfg.get("kl_warmup_min", 0) * 60 * spd)
    return cfg


def write_result(job, family, dataset, cfg, data, out, results_dir):
    """One self-contained file per finished job; the notebooks' reports and the final comparison read these.
    A search run writes its validation scores only (no test predictions)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    real = job.startswith("real_s")
    res = {"job": job, "family": "real" if real else family, "dataset": dataset, "stage": cfg["stage"],
           "tag": cfg.get("tag"), "seed": cfg["seed"], "data_hash": data["hash"], "feat_hash": data.get("feat_hash"),
           "classes": data["classes"], "train_counts": data["train_counts"].tolist(), "rare": data["rare"].tolist(),
           "config": {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, tuple, list, dict, type(None)))},
           "metrics": load_json(out / "metrics.json", [])}
    if not real and cfg["stage"] == "search":
        res.update(load_json(out / "search.json"))
        save_json(res, results_dir / f"{job}.json")
        return
    res.update(load_json(out / "final.json") or {"parts": load_json(out / "final_partial.json", {})})
    if real:
        res["stage"] = "final"
    save_json(res, results_dir / f"{job}.json")
    te = data["test_idx_np"]
    preds = dict(np.load(out / "final_pred.npz"))
    save_npz(results_dir / f"{job}_pred.npz", True, y_test=data["y_np"][te], star_test=data["star"][te], **preds)
    if (out / "samples.npz").exists():        # the synthetic curves, for the side-by-side figure and one-space fidelity
        shutil.copy2(out / "samples.npz", results_dir / f"{job}_samples.npz")


def speedtest(fam, cfg_over, data, device, n_steps=30, warm=10):
    """Training steps per second of one configuration (the batches, fp16 and EMA of a real run)."""
    cfg = dict(COMMON, **EVAL, **fam.DEFAULTS, seed=0, kl_warmup_its=0)
    cfg.update(cfg_over)
    torch.manual_seed(0)
    model = fam.MODEL(cfg, data, device)
    rs = ClassResampler(data["y_np"][data["train_idx_np"]], cfg["gamma"], seed=0, n_classes=data["n_classes"])
    draw = lambda: data["train_idx"][torch.from_numpy(rs.sample(cfg["batch"])).to(device)]
    sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else (lambda: None)
    for i in range(warm):
        model.step(draw, 10 ** 9)
    sync(); t = time.time()
    for i in range(n_steps):
        model.step(draw, 10 ** 9)
    sync()
    return n_steps / (time.time() - t), model


def benchmark(fam, data, device, quick=False):
    """SMOKE only: the steps per second of EVERY grid configuration alone on this GPU (they become the family's
    SPEED table), the sampling speed, and how long the classifier fits of an evaluation take.
    quick (my offline CPU test): fewer steps and curves, a quarter of a classifier epoch."""
    n_steps, n_sample, frac = (3, 256, 0.25) if quick else (40, 4096, 1.0)
    from .bench_eval import fit_classifiers
    speeds, model = {}, None
    for tag, over in fam.GRID.items():
        speeds[tag], m = speedtest(fam, over, data, device, n_steps=n_steps, warm=2 if quick else 10)
        model = model or m
    cfg = dict(COMMON, **EVAL, **fam.DEFAULTS, seed=0)
    sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else (lambda: None)
    out = {"speeds": speeds, "n_params": model.n_params()}
    idx = bench_data.balanced_idx(data, n_sample, seed=0)
    model.sample(idx[:64], seed=0)
    sync(); t = time.time()
    model.sample(idx, seed=0)
    sync()
    out["curves_per_s"] = len(idx) / (time.time() - t)
    tr, va, te = (bench_data.subset(data, data[f"{s}_idx"]) for s in ("train", "val", "test"))
    for runs in (cfg["sel_runs"], cfg["final_runs"]):
        t = time.time()
        fit_classifiers([tr], va, te, data["n_classes"], runs=runs, ens=cfg["ens"], max_epochs=frac, device=device,
                        log=lambda *a: None)
        out[f"clf_epoch_s_{runs * cfg['ens']}cnn"] = (time.time() - t) / frac
    out["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9 if str(device).startswith("cuda") else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--stage", choices=STAGES, default="search")
    ap.add_argument("--queue", default="")
    ap.add_argument("--deadline", type=float, default=1e12)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", default="catalina")
    ap.add_argument("--runs-dir", default=str(WORK / "runs"))
    ap.add_argument("--results-dir", default=str(WORK / "results"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--speedtest", type=int, default=0, help="only measure: this many steps of the default config")
    ap.add_argument("--speeds", default="", help="json file: grid key -> steps per second alone on a T4")
    a = ap.parse_args()
    runs_dir, results_dir = Path(a.runs_dir), Path(a.results_dir)
    tiny = a.smoke or os.environ.get("ASTRA_LOCAL_TEST") == "1"      # my offline CPU test runs tiny jobs
    dev = "cuda" if torch.cuda.is_available() else "cpu"              # cpu only in my offline tests
    deadline = Deadline(a.deadline)
    torch.backends.cudnn.benchmark = True
    fam = importlib.import_module(f"astra.bench_{a.family}")
    if a.speedtest:                       # SMOKE: is it worth running two jobs on one GPU?
        data = bench_data.load(a.data, 0, dev, "unit")
        sps, _ = speedtest(fam, {}, data, dev, n_steps=a.speedtest)
        print(json.dumps({"family": a.family, "it_per_s": sps}), flush=True)
        return
    for job in [j for j in a.queue.split(",") if j]:
        out = runs_dir / job
        status_path = out / "status.json"
        st = load_json(status_path, {}) or {}
        real = job.startswith("real_s")
        if st.get("state") == "done" and (real or st.get("stage") in ("final", a.stage)):
            print(f"[{job}] already done, skipping", flush=True)
            continue
        log = Logger(out / "log.txt")
        log(f"start {job}, {a.stage} stage ({'tiny' if tiny else 'real run'})")
        t0 = time.time()
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            speeds = json.loads(Path(a.speeds).read_text()) if a.speeds and Path(a.speeds).exists() else None
            cfg = job_config(job, fam, a.stage, tiny, speeds)
            save_json(cfg, out / "config.json")
            data = bench_data.load(a.data, cfg["seed"], dev, cfg["rep"])
            if real:
                state = "done" if run_real(data, cfg, out, deadline, dev, log) is not None else "paused"
            else:
                fid = Fidelity(data, feature_cnn(runs_dir, data, dev, 3 if tiny else 20, log), cfg["k"])
                state = BenchRun(cfg, data, fid, fam.MODEL, out, deadline, dev).run(reserve_s=900)
            if state == "done":
                write_result(job, a.family, a.dataset, cfg, data, out, results_dir)
            del data
        except Exception:
            log(traceback.format_exc())
            state = "failed"
        save_json({"state": state, "stage": "final" if real else a.stage, "hours_this_session": (time.time() - t0) / 3600,
                   "hours_total": st.get("hours_total", 0) + (time.time() - t0) / 3600,
                   "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0}, status_path)
        log(f"{job}: {state}")
        if state == "paused":
            break                 # out of time: the rest of the queue waits for the next session


if __name__ == "__main__":
    main()
