# ---- Launch the workers (SLOTS_PER_GPU per GPU) and wait ----
RESULTS_RESERVE_H = 0.5          # kept free for the results cells and Kaggle's output saving
deadline = RUN_STARTED + SESSION_HOURS * 3600 - RESULTS_RESERVE_H * 3600


def launch(queues, runs_dir, results_dir, smoke, stage):
    procs = {}
    for slot, q in queues.items():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot[0], PYTHONPATH=str(WORK))     # slot "1b" runs on GPU 1
        f = open(runs_dir / f"worker_{slot}.log", "a")
        cmd = [sys.executable, "-m", "astra.bench_run", "--family", FAMILY, "--stage", stage, "--queue", ",".join(q),
               "--deadline", str(deadline), "--data", str(NPZ), "--dataset", DATASET, "--runs-dir", str(runs_dir),
               "--results-dir", str(results_dir), "--speeds", str(SPEEDS_FILE)] + (["--smoke"] if smoke else [])
        procs[slot] = (subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=WORK), q)
    return procs


def state(job, runs_dir):
    p = runs_dir / job / "status.json"
    return json.loads(p.read_text()) if p.exists() else {}


def last_line(job, runs_dir):
    p = runs_dir / job / "log.txt"
    lines = p.read_text(encoding="utf-8").strip().splitlines() if p.exists() else []
    return lines[-1][:170] if lines else "(not started)"


def finished(job, runs_dir, stage):
    s = state(job, runs_dir)
    return s.get("state") == "failed" or (s.get("state") == "done" and (job.startswith("real_s") or s.get("stage") in ("final", stage)))


def wait(procs, runs_dir, every_s, stop_at, stage):
    """Returns as soon as every worker has exited; prints a status line per slot every `every_s` seconds.
    A worker pauses its job by itself 15 minutes before the deadline, so one that is still alive at `stop_at`
    is stuck: it is stopped here (its last saved state is kept) and the results cells still run."""
    t_print = time.time()
    while any(p.poll() is None for p, _ in procs.values()):
        time.sleep(5)
        if time.time() > stop_at:
            for slot, (p, _) in procs.items():
                if p.poll() is None:
                    p.kill()
                    p.wait()
                    print(f"!! the worker of slot {slot} was still running at the time limit and was stopped "
                          f"(see {runs_dir.name}/worker_{slot}.log); its last saved state is kept", flush=True)
        if all(p.poll() is not None for p, _ in procs.values()) or time.time() - t_print >= every_s:
            t_print = time.time()
            h = (time.time() - RUN_STARTED) / 3600
            for slot, (p, q) in procs.items():
                cur = next((j for j in q if not finished(j, runs_dir, stage)), q[-1])
                status = "running" if p.poll() is None else "exited "
                print(f"[{h:5.2f} h] slot {slot} {status} | {cur}: {last_line(cur, runs_dir)}", flush=True)
    return {slot: p.returncode for slot, (p, _) in procs.items()}


def run_stage(stage, queues, smoke):
    runs = STAGE_RUNS[stage]                 # final runs start fresh in their own folder (since 2026-10-22)
    runs.mkdir(parents=True, exist_ok=True)
    for f in RUNS.glob("feature_cnn_s*"):    # SMOKE runs both stages: the final folder needs the feature CNNs too
        if not (runs / f.name).exists():
            shutil.copy2(f, runs / f.name)
    todo = {s: [j for j in q if not finished(j, runs, stage)] for s, q in queues.items()}
    todo = {s: q for s, q in todo.items() if q}
    if not todo:
        print(f"{stage} stage: every job is already done")
        return {}
    if not smoke:      # canary: a tiny version of each slot's first model job, in a throw-away folder
        canary = WORK / "canary"
        shutil.rmtree(canary, ignore_errors=True)
        canary.mkdir()
        for f in runs.glob("feature_cnn_s*"):
            shutil.copy2(f, canary / f.name)
        first = {s: [next(j for j in q if not j.startswith("real_s"))] for s, q in todo.items()
                 if any(not j.startswith("real_s") for j in q)}
        codes = wait(launch(first, canary, canary / "results", True, "search"), canary, 20, time.time() + 1800, "search")
        states = {j: state(j, canary).get("state") for q in first.values() for j in q}
        assert all(c == 0 for c in codes.values()) and all(s == "done" for s in states.values()), \
            f"canary failed: {codes} {states} (see {canary})"
        shutil.rmtree(canary)
        print("canary passed\n")
    codes = wait(launch(todo, runs, RESULTS, smoke, stage), runs, 30 if smoke else 600, deadline, stage)
    print(f"{stage} stage: workers finished with exit codes", codes)
    return codes


# ---- how many jobs share a GPU: measured once (SMOKE cell or here), kept in runs_*/packing.json ----
PACK_FILE = RUNS / "packing.json"
if SLOTS_PER_GPU != "auto":
    SPG = int(SLOTS_PER_GPU)
elif LOCAL_TEST:
    SPG = 1
elif PACK_FILE.exists():
    SPG = json.loads(PACK_FILE.read_text())["recommended"]
else:
    pk = (json.load(open(TABLES / "benchmark.json")).get("packing") if (TABLES / "benchmark.json").exists() else None)
    if not pk:
        solo, pair = packing_test(1, ["0"], 300), packing_test(2, ["0"], 300)
        gain = sum(pair) / solo[0] if solo and pair else 1.0
        pk = {"solo": solo, "pair": pair, "gain2": gain, "recommended": 2 if gain >= 1.25 else 1}
    PACK_FILE.write_text(json.dumps(pk, indent=1))
    SPG = pk["recommended"]
print(f"jobs per GPU: {SPG}" + ("" if SLOTS_PER_GPU != "auto" else f" (auto; {PACK_FILE.name})"))
if not SMOKE and not LOCAL_TEST:   # fails here, in minutes, if a job has no speed (instead of in every worker)
    for j in JOBS_ALL:
        if not j.startswith("real_s"):
            bench_run.job_config(j, fam, STAGE, False, SPEEDS)

if SMOKE:              # a tiny search, then a tiny final stage (fresh runs of the first configuration, splits 0 and 1)
    run_stage("search", make_queues(SEARCH_JOBS, SPG), True)
    smoke_final = [SEARCH_JOBS[0], SEARCH_JOBS[0][:-1] + "1"]
    smoke_jobs = smoke_final + ["real_s0"]
    run_stage("final", make_queues(smoke_jobs, SPG), True)
    SHOWN = [(j, STAGE_RUNS["search"]) for j in SEARCH_JOBS] + [(j, STAGE_RUNS["final"]) for j in smoke_jobs]
else:
    run_stage(STAGE, make_queues(JOBS_ALL, SPG), False)
    SHOWN = [(j, RUNS) for j in JOBS_ALL]
for j, runs in SHOWN:                   # (job, its runs folder): in SMOKE the first job runs in both stages
    print(f"  {runs.name + '/' + j:<44} {state(j, runs)}")
