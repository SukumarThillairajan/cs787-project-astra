"""Execute the data-audit notebook (ASTRA_00) offline, the way Kaggle would.

    python build/tests/run_local_data.py [out_dir]

Env: ASTRA_SCRATCH (work folder), ASTRA_RAW (folder with catalina_L100.npz and/or ztf_L40.npz; optionally
CatalinaVars.tbl, AllVar.phot.gz and the PELS-VAE .npy.gz file, which then are not downloaded).
If out_dir is given, the figures and tables are copied there (e.g. report/data_audit).
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

sys.stdout.reconfigure(encoding="utf-8")
SCR = Path(os.environ["ASTRA_SCRATCH"])
work, inp = SCR / "data_audit_work", SCR / "data_audit_input"
shutil.rmtree(work, ignore_errors=True)
shutil.rmtree(inp, ignore_errors=True)
work.mkdir(parents=True)
(inp / "raw").mkdir(parents=True)
for f in Path(os.environ["ASTRA_RAW"]).iterdir():
    os.link(f, inp / "raw" / f.name)
os.environ.update(ASTRA_WORK=str(work), ASTRA_INPUT=str(inp), PYTHONIOENCODING="utf-8")

name = "ASTRA_00_DATA_audit.ipynb"
nb = build(name, *NOTEBOOKS[name])
t = time.time()
client = NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(work)}})
try:
    client.execute()
    status = "OK"
except Exception as exc:
    status = f"FAILED: {type(exc).__name__}: {str(exc)[-3000:]}"
nbformat.write(nb, SCR / "executed_data_audit.ipynb")
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
            print(f"--- cell {i} ---\n{txt[:8000]}")
everything = "\n".join(text)
if status == "OK" and "!!" in everything:
    status = "FAILED: a table or figure reported '!!'"
if status == "OK" and "ALL DELIVERABLES PRESENT" not in everything:
    status = "FAILED: deliverables missing"
if status == "OK" and len(sys.argv) > 1:
    out = Path(sys.argv[1])
    for sub in ("figures_data", "tables_data"):
        shutil.copytree(work / sub, out / sub.replace("_data", ""), dirs_exist_ok=True)
    print("copied figures and tables to", out)
print(f"\ndata audit: {status} in {time.time() - t:.0f}s")
