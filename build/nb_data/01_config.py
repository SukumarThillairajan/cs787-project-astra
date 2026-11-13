# ---- Configuration (CPU only) ----
import os, sys, json, time, shutil, traceback
from pathlib import Path

DATASETS = ["catalina", "ztf"]      # ztf is skipped if ztf_L40.npz is not attached under Add Input
WITH_OGLE = True                    # section 6: the PELS-VAE training file (296 MB download)
WITH_PIPELINE = True                # section 2: the raw-to-input figure (needs the raw Catalina photometry, 271 MB)
SEED = 0                            # the split shown is the benchmark's split 0
L = {"catalina": 100, "ztf": 40}
CLASS_RANGE_K, DATA_SEED = 2.0, 0   # Catalina pre-processing, as in notebook 01

RUN_STARTED = time.time()
WORK = Path(os.environ.get("ASTRA_WORK", "/kaggle/working"))
INPUT = Path(os.environ.get("ASTRA_INPUT", "/kaggle/input"))
FIGS, TABLES = WORK / "figures_data", WORK / "tables_data"
for d in (FIGS, TABLES, WORK / "data", WORK / "astra"):
    d.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(WORK))


def safely(what, fn):
    """One failing figure must not stop the rest of the audit: report it and go on."""
    try:
        return fn()
    except Exception:
        print(f"\n!! {what} failed:\n{traceback.format_exc()}")


def show(title, table, name, index=False):
    if table is not None and len(table):
        table.to_csv(TABLES / name, index=index)
        print(f"\n{title}\n{table.to_string(index=index)}")
