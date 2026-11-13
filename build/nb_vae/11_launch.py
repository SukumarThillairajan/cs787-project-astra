# ---- Launch the two GPU workers and wait ----
del lcs                                  # the workers load their own copy
RESULTS_RESERVE_H = 0.4                  # kept free for the regressions, figures and output saving
deadline = RUN_STARTED + SESSION_HOURS * 3600 - RESULTS_RESERVE_H * 3600


def launch(queues, runs_dir, smoke):
    procs = {}
    for gpu, q in queues.items():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(WORK))
        f = open(runs_dir / f"worker_gpu{gpu}.log", "a")
        cmd = [sys.executable, "-m", "astra.run_vae", "--queue", ",".join(q), "--deadline", str(deadline),
               "--runs-dir", str(runs_dir)] + (["--smoke"] if smoke else [])
        procs[gpu] = (subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=WORK), q)
    return procs


def last_line(job, runs_dir):
    p = runs_dir / job / "log.txt"
    lines = p.read_text(encoding="utf-8").strip().splitlines() if p.exists() else []
    return lines[-1][:150] if lines else "(not started)"


def job_state(job, runs_dir):
    p = runs_dir / job / "status.json"
    return json.loads(p.read_text())["state"] if p.exists() else ""


def wait(procs, runs_dir, every_s, stop_at):
    """Returns as soon as every worker has exited and prints a status line every `every_s` seconds.
    A worker pauses its job by itself 15 minutes before the deadline, so one that is still alive at `stop_at`
    is stuck: it is stopped here (its last saved epoch is kept) and the session goes on to the results cells
    instead of running into Kaggle's 12-hour limit, where the whole output would be lost."""
    t_print = time.time()
    while any(p.poll() is None for p, _ in procs.values()):
        time.sleep(5)
        if time.time() > stop_at:
            for gpu, (p, _) in procs.items():
                if p.poll() is None:
                    p.kill()
                    p.wait()
                    print(f"!! the worker on GPU {gpu} was still running at the time limit and was stopped "
                          f"(see {runs_dir.name}/worker_gpu{gpu}.log); its last saved state is kept", flush=True)
        over = all(p.poll() is not None for p, _ in procs.values())
        if over or time.time() - t_print >= every_s:
            t_print = time.time()
            h = (time.time() - RUN_STARTED) / 3600
            for gpu, (p, q) in procs.items():
                cur = next((j for j in q if job_state(j, runs_dir) not in ("done", "failed")), q[-1])
                print(f"[{h:5.2f} h] GPU {gpu} {'running' if p.poll() is None else 'exited '} | {cur}: "
                      f"{last_line(cur, runs_dir)}", flush=True)
    return {gpu: p.returncode for gpu, (p, _) in procs.items()}


todo = [j for q in QUEUES.values() for j in q if job_state(j, RUNS) != "done"]
if not todo:           # a results-only session: no canary, no workers
    print("every job is already done: nothing to train; the results cells use the restored files")
else:
    if not SMOKE:      # canary: tiny versions of each GPU's first job, in a throw-away folder
        canary = WORK / "canary"
        canary.mkdir(exist_ok=True)
        codes = wait(launch({g: q[:1] for g, q in QUEUES.items()}, canary, True), canary, 20, time.time() + 900)
        states = {j.name: json.loads((j / "status.json").read_text())["state"] for j in canary.iterdir()
                  if (j / "status.json").exists()}
        assert all(c == 0 for c in codes.values()) and all(s == "done" for s in states.values()), \
            f"canary failed: {codes} {states} (see {canary})"
        shutil.rmtree(canary)
        print("canary passed\n")
    print(f"jobs to run: {todo} | the workers pause their jobs by {(deadline - 900 - RUN_STARTED) / 3600:.2f} h "
          f"and are stopped at {(deadline - RUN_STARTED) / 3600:.2f} h at the latest")
    codes = wait(launch(QUEUES, RUNS, SMOKE), RUNS, 30 if SMOKE else 600, deadline)
    print("workers finished with exit codes", codes)
for p in sorted(RUNS.glob("*/status.json")):
    print(f"  {p.parent.name:<14} {json.loads(p.read_text())}")
