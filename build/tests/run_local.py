"""Execute a notebook offline (CPU) the way Kaggle would, with small patches for speed.

    python build/tests/run_local.py gan smoke        # SMOKE = True
    python build/tests/run_local.py gan real         # SMOKE = False, tiny jobs (ASTRA_LOCAL_TEST=1)
    python build/tests/run_local.py gan real2        # a second session on top of `real` (resume/skip path)
    python build/tests/run_local.py vae smoke|real|real2
    python build/tests/run_local.py ganztf smoke     # the GAN notebook with DATASET = "ztf" (ASTRA_RAW holds ztf_L40.npz)

Env: ASTRA_SCRATCH (bulky outputs), ASTRA_RAW (folder with the raw data files, attached as if a Kaggle input).
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

sys.stdout.reconfigure(encoding="utf-8")         # notebook outputs contain characters like "≈"
which, mode = sys.argv[1], sys.argv[2]
SCR = Path(os.environ["ASTRA_SCRATCH"])
work, inp = SCR / f"{which}_{mode}_work", SCR / f"{which}_{mode}_input"
shutil.rmtree(work, ignore_errors=True)
shutil.rmtree(inp, ignore_errors=True)
work.mkdir(parents=True)
(inp / "raw").mkdir(parents=True)
for f in Path(os.environ["ASTRA_RAW"]).iterdir():               # "attached dataset" with the raw files
    os.link(f, inp / "raw" / f.name) if not (inp / "raw" / f.name).exists() else None
if mode == "real2":                                              # previous session's output attached
    shutil.copytree(SCR / f"{which}_real_work", inp / "prev_output", ignore=shutil.ignore_patterns("astra"))
os.environ.update(ASTRA_WORK=str(work), ASTRA_INPUT=str(inp), ASTRA_LOCAL_TEST="1", PYTHONIOENCODING="utf-8")

name = next(n for n in NOTEBOOKS if which[:3].upper() in n)
folder, order = NOTEBOOKS[name]
nb = build(name, folder, order)
NL = chr(10)
GAN_QUEUES = (NL.join(['QUEUES = {', '    "0": ["gan_gamma025_s0", "gan_gamma025_s1", "gan_gamma0_s0"],',
                       '    "1": ["trtr", "gan_u_s0", "gan_gamma025_s2"],', '}']),
              'QUEUES = {"0": ["gan_gamma025_s0"], "1": ["trtr", "gan_u_s0"]}')
VAE_QUEUES = (NL.join(['QUEUES = {', '    "0": ["tcn", "gru", "tcn_s14"],', '    "1": ["tcn_phys", "tcn_phys_s14"],', '}']),
              'QUEUES = {"0": ["tcn", "gru"], "1": ["tcn_phys", "tcn_s14"]}')
# the SMOKE benchmark times real-size evaluations, far too slow on a CPU: shrink them (same code path)
BENCH = ("    from astra import run_gan" + NL,
         "    from astra import run_gan" + NL +
         "    run_gan.GAN.update(final_runs=2, dr_reps=1, fid_batches=2, fid_batch_size=500)   # offline test only" + NL)
patch = {"gan": [GAN_QUEUES],
         "ganztf": [GAN_QUEUES, ('DATASET = "catalina" ', 'DATASET = "ztf"      ')],
         "vae": [VAE_QUEUES]}[which]
if which.startswith("gan"):
    patch.append(BENCH)
if mode != "smoke":
    patch.append(("SMOKE = True ", "SMOKE = False"))
if mode == "real2":
    patch.append(("SESSION = 1 ", "SESSION = 2 "))
for old, new in patch:                           # every patch must hit exactly one cell
    hits = [c for c in nb.cells if c.cell_type == "code" and not c.source.startswith("%%writefile") and old in c.source]
    assert len(hits) == 1, f"patch {old[:40]!r} matched {len(hits)} cells"
    hits[0].source = hits[0].source.replace(old, new)
t = time.time()
client = NotebookClient(nb, timeout=14400, kernel_name="python3", resources={"metadata": {"path": str(work)}})
try:
    client.execute()
    status = "OK"
except Exception as exc:
    status = f"FAILED: {type(exc).__name__}: {str(exc)[-3000:]}"
nbformat.write(nb, SCR / f"executed_{which}_{mode}.ipynb")
for i, c in enumerate(nb.cells):
    if c.cell_type != "code" or c.source.startswith("%%writefile"):
        continue
    for o in c.get("outputs", []):
        txt = o.get("text") or "".join(o.get("data", {}).get("text/plain", "")) or ""
        if o.get("output_type") == "error":
            txt = "ERROR " + o["ename"] + ": " + o["evalue"]
        if txt.strip():
            print(f"--- cell {i} ---{NL}{txt[:3000]}")
print(f"{NL}{which} {mode}: {status} in {time.time() - t:.0f}s")
