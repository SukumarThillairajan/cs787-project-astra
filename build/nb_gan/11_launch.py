# ---- Launch the two GPU workers and wait ----
RESULTS_RESERVE_H = 0.5          # kept free for the results cells and Kaggle's output saving
deadline = RUN_STARTED + SESSION_HOURS * 3600 - RESULTS_RESERVE_H * 3600


def launch(queues, runs_dir, smoke):
    procs = {}
    for gpu, q in queues.items():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(WORK))
        f = open(runs_dir / f"worker_gpu{gpu}.log", "a")
        cmd = [sys.executable, "-m", "astra.run_gan", "--queue", ",".join(q), "--deadline", str(deadline),
               "--runs-dir", str(runs_dir), "--data", str(NPZ), "--latent", str(LATENT)] + (["--smoke"] if smoke else [])
        procs[gpu] = (subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=WORK), q)
    return procs


def last_line(job, runs_dir):
    p = runs_dir / job / "log.txt"
    lines = p.read_text(encoding="utf-8").strip().splitlines() if p.exists() else []
    return lines[-1][:150] if lines else "(not started)"


def wait(procs, runs_dir, every_s):
    while any(p.poll() is None for p, _ in procs.values()):
        time.sleep(every_s)
        h = (time.time() - RUN_STARTED) / 3600
        for gpu, (p, q) in procs.items():
            cur = next((j for j in q if (json.loads((runs_dir / j / "status.json").read_text())["state"]
                        if (runs_dir / j / "status.json").exists() else "") not in ("done", "failed")), q[-1])
            print(f"[{h:5.2f} h] GPU {gpu} {'running' if p.poll() is None else 'exited '} | {cur}: "
                  f"{last_line(cur, runs_dir)}", flush=True)
    return {gpu: p.returncode for gpu, (p, _) in procs.items()}


if not SMOKE:          # canary: a tiny version of each GPU's first job, in a throw-away folder
    canary = WORK / "canary"
    canary.mkdir(exist_ok=True)
    shutil.copy2(RUNS / "feature_cnn.pt", canary / "feature_cnn.pt")
    codes = wait(launch({g: [next(j for j in q if j != "trtr")] for g, q in QUEUES.items()}, canary, True), canary, 20)
    states = {j.name: json.loads((j / "status.json").read_text())["state"] for j in canary.iterdir()
              if (j / "status.json").exists()}
    assert all(c == 0 for c in codes.values()) and all(s == "done" for s in states.values()), \
        f"canary failed: {codes} {states} (see {canary})"
    shutil.rmtree(canary)
    print("canary passed\n")

codes = wait(launch(QUEUES, RUNS, SMOKE), RUNS, 30 if SMOKE else 600)
print("workers finished with exit codes", codes)
for p in sorted(RUNS.glob("*/status.json")):
    print(f"  {p.parent.name:<18} {json.loads(p.read_text())}")
