# ---- SMOKE only: real-size speeds -> the family's SPEED table, and whether two jobs per GPU pay off ----
def packing_test(n_proc_per_gpu, gpus, steps):
    """Training steps per second of the default configuration when n processes share each GPU."""
    procs = []
    for g in gpus:
        for _ in range(n_proc_per_gpu):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONPATH=str(WORK))
            cmd = [sys.executable, "-m", "astra.bench_run", "--family", FAMILY, "--data", str(NPZ),
                   "--speedtest", str(steps)]
            procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=WORK, text=True))
    outs = [p.communicate()[0] for p in procs]
    try:
        return [json.loads(o.strip().splitlines()[-1])["it_per_s"] for o in outs]
    except Exception:
        print("!! packing test failed:\n", "\n".join(o[-1500:] for o in outs))
        return None


if SMOKE:
    data = bench_data.load(NPZ, 0, DEV)
    b = bench_run.benchmark(fam, data, DEV, quick=LOCAL_TEST)
    n_train = len(data["train_idx"])
    del data
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items() if k != "speeds"})
    print(f"\n{FAMILY}: training steps per second ALONE on one GPU, per grid configuration "
          f"(for information only: the search measures its own speeds; nothing to paste):")
    print("SPEED = {" + ", ".join(f'"{k}": {v:.1f}' for k, v in b["speeds"].items()) + "}")
    for k, v in b["speeds"].items():
        bs = {**fam.DEFAULTS, **fam.GRID[k]}["batch"]
        print(f"  {k:<14} {v:6.1f} steps/s -> 1 h = {v * 3600:,.0f} steps = {v * 3600 * bs / n_train:,.0f} passes "
              f"over the {n_train:,} training curves")
    steps = 20 if LOCAL_TEST else 300
    if not LOCAL_TEST:
        solo = packing_test(1, ["0"], steps)
        pair = packing_test(2, ["0"], steps)
        quad = packing_test(2, ["0", "1"], steps)
        if solo and pair and quad:
            gain2, gain4 = sum(pair) / solo[0], sum(quad) / (2 * solo[0])
            rec = 2 if min(gain2, gain4) >= 1.25 else 1
            print(f"\npacking test ({steps} steps each): alone {solo[0]:.1f} steps/s | two on one GPU "
                  f"{' + '.join(f'{v:.1f}' for v in pair)} = {gain2:.2f} x | two on each GPU, per GPU {gain4:.2f} x"
                  f"\n-> recommended SLOTS_PER_GPU = {rec} (2 only if sharing gives at least 1.25 x the work per GPU)")
            b["packing"] = {"solo": solo, "pair": pair, "quad": quad, "gain2": gain2, "gain4": gain4, "recommended": rec}
    E = bench_eval.EVAL
    clf_sel = 4 * b[f"clf_epoch_s_{E['sel_runs'] * E['ens']}cnn"] * E["n_sel"] / n_train
    eval_min = ((E["n_sel"] + 2 * E["n_mem"]) / b["curves_per_s"] + clf_sel + 20) / 60
    print(f"\nsampling: {b['curves_per_s']:,.0f} curves per second with the default settings | one checkpoint "
          f"evaluation ≈ {eval_min:.1f} min alone on a GPU | peak GPU memory {b['peak_gpu_gb']:.1f} GB")
    json.dump(b, open(TABLES / "benchmark.json", "w"), indent=1, default=str)

# ---- every real run: the speeds of its configurations (kept in runs_*/speeds.json for later sessions) ----
SPEEDS_FILE = RUNS / "speeds.json"
SPEEDS = json.loads(SPEEDS_FILE.read_text()) if SPEEDS_FILE.exists() else {}
need = sorted({bench_run.job_tag(j, FAMILY) for j in JOBS_ALL if not j.startswith("real_s")}
              - {t for t, v in fam.SPEED.items() if v} - set(SPEEDS))
if not SMOKE and not LOCAL_TEST and need:
    if STAGE == "final":
        raise AssertionError(f"no speed for {need}: attach the search output of this configuration, or put its "
                             "speed into astra/bench_grid.py and rebuild")
    data = bench_data.load(NPZ, 0, DEV)
    for t in need:
        SPEEDS[t], _ = bench_run.speedtest(fam, fam.GRID[t], data, DEV, n_steps=60)
    del data
    SPEEDS_FILE.write_text(json.dumps(SPEEDS, indent=1))
    print("measured now (alone on this GPU, kept for the next sessions):")
if SPEEDS:
    print(f"speeds in {SPEEDS_FILE.name}: SPEED[\"{FAMILY}\"] = " + json.dumps({k: round(v, 2) for k, v in SPEEDS.items()}))
