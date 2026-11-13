"""Execute a benchmark notebook offline (CPU) the way Kaggle would, with tiny jobs.

    python build/tests/run_local_bench.py gan smoke catalina         # SMOKE = True (tiny search + tiny final)
    python build/tests/run_local_bench.py fm real catalina           # search stage, two jobs, tiny (ASTRA_LOCAL_TEST=1)
    python build/tests/run_local_bench.py fm real2 catalina gan,vae  # final stage with `real`'s search output
                                                                     # attached (it must be ignored): fresh runs of
                                                                     # the first configuration on splits 0, 1 and
                                                                     # real_s0-1; with the `real2` outputs of gan and
                                                                     # vae attached, which exercises the comparison
    python build/tests/run_local_bench.py fm real3 catalina          # a final session 2 on top of `real2`: restores
                                                                     # the final folder, every job is already done

Env: ASTRA_SCRATCH (bulky outputs), ASTRA_RAW (folder with catalina_L100.npz and/or ztf_L40.npz, attached as if a
Kaggle input). The run fails if a cell raises, a table or figure reports "!!", or a deliverable is missing.
"""
import os
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_notebooks import NOTEBOOKS, build  # noqa: E402
from astra.bench_grid import search_jobs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
fam, mode, dataset = sys.argv[1], sys.argv[2], sys.argv[3]
attach = sys.argv[4].split(",") if len(sys.argv) > 4 else []
SCR = Path(os.environ["ASTRA_SCRATCH"])
tag = f"bench_{fam}_{dataset}"
work, inp = SCR / f"{tag}_{mode}_work", SCR / f"{tag}_{mode}_input"
shutil.rmtree(work, ignore_errors=True)
shutil.rmtree(inp, ignore_errors=True)
work.mkdir(parents=True)
(inp / "raw").mkdir(parents=True)
for f in Path(os.environ["ASTRA_RAW"]).iterdir():               # "attached dataset" with the data files
    os.link(f, inp / "raw" / f.name)
skip = shutil.ignore_patterns("astra", "*.pt", "canary")
if mode in ("real2", "real3"):                                   # this notebook's previous output (weights included)
    prev = {"real2": "real", "real3": "real2"}[mode]
    shutil.copytree(SCR / f"{tag}_{prev}_work", inp / "prev_output", ignore=shutil.ignore_patterns("astra"))
for other in attach:                                             # other benchmark notebooks' outputs
    shutil.copytree(SCR / f"bench_{other}_{dataset}_real2_work", inp / f"output_{other}", ignore=skip)
os.environ.update(ASTRA_WORK=str(work), ASTRA_INPUT=str(inp), ASTRA_LOCAL_TEST="1", PYTHONIOENCODING="utf-8")

name = next(n for n in NOTEBOOKS if f"BENCH_{fam.upper()}" in n)
nb = build(name, *NOTEBOOKS[name])
cfg = next(c for c in nb.cells if c.cell_type == "code" and c.source.startswith("# ---- Configuration"))
lines = cfg.source.split("\n")
two = search_jobs(fam)[:2]
tags = [j[len(fam) + 1:j.rindex("_s")] for j in two]
patch = {'DATASET = "catalina"': f'DATASET = "{dataset}"    # offline test'}
if mode != "smoke":
    patch["SMOKE = True"] = "SMOKE = False"
    patch["SEARCH_CONFIGS = "] = f"SEARCH_CONFIGS = {tags!r}"
if mode in ("real2", "real3"):
    patch["SESSION = 1"] = f"SESSION = {mode[-1]}"
    patch['STAGE = "search"'] = 'STAGE = "final"'
    patch["FINAL_TAG = "] = f"FINAL_TAG = {tags[0]!r}"
    patch["FINAL_SPLITS = "] = "FINAL_SPLITS = [0, 1]"
for start, new in patch.items():                 # every patch must hit exactly one line of the configuration cell
    hits = [i for i, line in enumerate(lines) if line.startswith(start)]
    assert len(hits) == 1, f"patch {start!r} matched {len(hits)} lines"
    lines[hits[0]] = new
cfg.source = "\n".join(lines)

t = time.time()
client = NotebookClient(nb, timeout=14400, kernel_name="python3", resources={"metadata": {"path": str(work)}})
try:
    client.execute()
    status = "OK"
except Exception as exc:
    status = f"FAILED: {type(exc).__name__}: {str(exc)[-3000:]}"
nbformat.write(nb, SCR / f"executed_{tag}_{mode}.ipynb")
text = []
for i, c in enumerate(nb.cells):
    if c.cell_type != "code" or c.source.startswith("%%writefile"):
        continue
    for o in c.get("outputs", []):
        txt = o.get("text") or "".join(o.get("data", {}).get("text/plain", "")) or ""
        if o.get("output_type") == "error":
            txt = "ERROR " + o["ename"] + ": " + o["evalue"]
        if txt.strip():
            text.append(txt)
            print(f"--- cell {i} ---\n{txt[:6000]}")
everything = "\n".join(text)
if status == "OK" and "!!" in everything:
    status = "FAILED: a table or figure reported '!!'"
if status == "OK" and "ALL DELIVERABLES PRESENT" not in everything:
    status = "FAILED: deliverables missing"
if status == "OK" and attach and "comparison with attached results" not in everything:
    status = "FAILED: the comparison with the attached outputs did not run"
if status == "OK" and mode == "real2" and "fresh start (nothing to restore)" not in everything:
    status = "FAILED: the final stage restored files from the attached search output"
if status == "OK" and mode == "real3" and not ("restored " in everything and "every job is already done" in everything):
    status = "FAILED: the final session 2 did not restore and skip the finished final jobs"
if status == "OK" and mode in ("real2", "smoke") and "novel_selected" not in everything + "".join(
        p.read_text(encoding="utf-8", errors="replace") for p in work.rglob("final.json")):
    status = "FAILED: no new-condition scores in the final results"
print(f"\nbench {fam} {mode} {dataset}: {status} in {time.time() - t:.0f}s")
