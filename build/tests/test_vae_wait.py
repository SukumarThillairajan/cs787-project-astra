"""The launch cell of the VAE notebook must not wait for ever: a worker that is still alive at the time limit
is stopped, and wait() returns as soon as every worker has exited. No torch needed.

    python build/tests/test_vae_wait.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

src = (Path(__file__).resolve().parents[1] / "nb_vae" / "11_launch.py").read_text(encoding="utf-8")
body = src[src.index("def launch"):src.index("todo = ")]            # the helper functions of the cell only
ns = dict(json=json, os=os, sys=sys, subprocess=subprocess, time=time, RUN_STARTED=time.time())
exec(body, ns)
wait = ns["wait"]
runs = Path(tempfile.mkdtemp())
sleeper = lambda s: subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({s})"])

# 1. a stuck worker (sleeps 10 minutes) next to a healthy one: stopped at the limit, the other's code is 0
t = time.time()
codes = wait({"0": (sleeper(600), ["job_a"]), "1": (sleeper(1), ["job_b"])}, runs, 600, time.time() + 8)
took = time.time() - t
assert codes["1"] == 0 and codes["0"] not in (0, None), codes
assert 8 <= took < 30, took
print(f"stuck worker stopped after {took:.0f} s, exit codes {codes}")

# 2. healthy workers: returns right after the last one exits, long before the print interval or the limit
t = time.time()
codes = wait({"0": (sleeper(2), ["job_a"]), "1": (sleeper(4), ["job_b"])}, runs, 600, time.time() + 3600)
took = time.time() - t
assert codes == {"0": 0, "1": 0} and took < 20, (codes, took)
print(f"healthy workers: returned after {took:.0f} s, exit codes {codes}")

# 3. the status line names the first job of the queue that is not done
(runs / "job_a").mkdir()
(runs / "job_a" / "status.json").write_text(json.dumps({"state": "done"}))
assert ns["job_state"]("job_a", runs) == "done" and ns["job_state"]("job_b", runs) == ""
print("VAE WAIT TEST PASSED")
