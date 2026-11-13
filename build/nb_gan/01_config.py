# ---- Configuration ----
import os, sys, json, time, shutil, subprocess
from pathlib import Path

SMOKE = True            # True: tests + tiny jobs + timing (run interactively first); False: the real run
DATASET = "catalina"    # "catalina", or "ztf" (then attach the dataset that holds ztf_L40.npz under Add Input)
SESSION = 1             # 1, 2, 3, ... only labels this run's summary
SESSION_HOURS = 10.5    # stop cleanly before Kaggle's 12-hour limit; lower it if the account has less quota left

# Each GPU works down its own queue; unfinished jobs resume in the next session (previous output attached).
# trtr = rows A-D, gan_u = u-GAN (rows E, F), gan_gamma025 = gamma-GAN (rows G, H); s0/s1/s2 = seeds
QUEUES = {
    "0": ["gan_gamma025_s0", "gan_gamma025_s1", "gan_gamma0_s0"],
    "1": ["trtr", "gan_u_s0", "gan_gamma025_s2"],
}
L, LATENT = {"catalina": (100, 16), "ztf": (40, 8)}[DATASET]    # points per curve, noise size (Sec. 3.2.3, 4.2)
CLASS_RANGE_K = 2.0     # class-range outlier filter width (not given in the paper; 2 sigma matches Table 2)
DATA_SEED = 0

LOCAL_TEST = os.environ.get("ASTRA_LOCAL_TEST") == "1"     # only my offline test run sets this
RUN_STARTED = time.time()
WORK = Path(os.environ.get("ASTRA_WORK", "/kaggle/working"))
INPUT = Path(os.environ.get("ASTRA_INPUT", "/kaggle/input"))
RUNS = WORK / (f"runs_{DATASET}" + ("_smoke" if SMOKE else ""))      # smoke outputs never mix with the real ones
FIGS, TABLES = WORK / f"figures_{DATASET}", WORK / f"tables_{DATASET}"
for d in (RUNS, FIGS, TABLES, WORK / "data", WORK / "astra"):
    d.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(WORK))
print(f"{DATASET}: session {SESSION} ({SESSION_HOURS} h), {'SMOKE' if SMOKE else 'REAL RUN'} | queues: {QUEUES}")
