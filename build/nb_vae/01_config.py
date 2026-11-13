# ---- Configuration ----
import os, sys, json, time, shutil, subprocess
from pathlib import Path

SMOKE = True            # True: tests + tiny jobs + timing (run interactively first); False: the real run
SESSION = 1             # 1, 2, ... only labels this run's summary
SESSION_HOURS = 10.5    # time limit of the training (jobs pause ~40 min before it); it does not end the notebook

# Each GPU trains its queue one model after another (tcn = their best cVAE, tcn_phys = cVAE-P,
# gru = the GRU of Fig. 4, *_s14 = a second seed with the same split)
QUEUES = {
    "0": ["tcn", "gru", "tcn_s14"],
    "1": ["tcn_phys", "tcn_phys_s14"],
}

LOCAL_TEST = os.environ.get("ASTRA_LOCAL_TEST") == "1"     # only my offline test run sets this
RUN_STARTED = time.time()
WORK = Path(os.environ.get("ASTRA_WORK", "/kaggle/working"))
INPUT = Path(os.environ.get("ASTRA_INPUT", "/kaggle/input"))
RUNS = WORK / ("runs_smoke" if SMOKE else "runs")        # smoke outputs never mix with the real ones
FIGS, TABLES = WORK / "figures", WORK / "tables"
for d in (RUNS, FIGS, TABLES, WORK / "data", WORK / "astra"):
    d.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(WORK))
print(f"session {SESSION} ({SESSION_HOURS} h), {'SMOKE' if SMOKE else 'REAL RUN'} | queues: {QUEUES}")
