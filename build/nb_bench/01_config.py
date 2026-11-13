# ---- Configuration ----
import os, sys, json, time, shutil, subprocess
from pathlib import Path

FAMILY = "@@FAMILY@@"   # this notebook's model family (astra/bench_@@FAMILY@@.py)
SMOKE = True            # True: tests, tiny search and final jobs, speeds (run first); False: a real run
STAGE = "search"        # "search": every grid configuration on split 0 (validation only); "final": the winner on
                        # FINAL_SPLITS with test scores (fresh runs: no search output needed)
DATASET = "catalina"    # the benchmark uses Catalina only (user decision of 2026-10-15)
SESSION = 1             # 1, 2, 3, ... only labels this run's summary
SESSION_HOURS = 10.5    # time limit of the training (jobs pause ~25 min before it); lower it if the quota is short
SLOTS_PER_GPU = "auto"  # jobs sharing one GPU: "auto" = measured in the first session (2 if it pays), or 1 / 2
SEARCH_CONFIGS = "all"  # "all", or a list of grid keys (to split one family's search over two accounts)
FINAL_SPLITS = [0, 1, 2, 3, 4]   # final stage: the splits this account runs (to split them over accounts)

# Jobs (the grid, the winner and the speeds are in astra/bench_grid.py). Search: one job per grid configuration on
# split 0. Final: the winner on FINAL_SPLITS plus the reference rows real_s<k> (A = oversampling, B-D = warping).
GRID_TAGS = @@GRID_TAGS@@
FINAL_TAG = @@FINAL_TAG@@
SEARCH_JOBS = [f"{FAMILY}_{t}_s0" for t in (GRID_TAGS if SEARCH_CONFIGS == "all" else SEARCH_CONFIGS)]
FINAL_JOBS = [f"{FAMILY}_{FINAL_TAG}_s{k}" for k in FINAL_SPLITS] if FINAL_TAG else []
JOBS_ALL = SEARCH_JOBS if STAGE == "search" else FINAL_JOBS + [f"real_s{k}" for k in FINAL_SPLITS]


def make_queues(jobs, slots_per_gpu):
    """Slots 0a, 0b run on GPU 0 and 1a, 1b on GPU 1; jobs are dealt out in turn (the longest are listed first)."""
    slots = [f"{g}{chr(97 + k)}" for g in "01" for k in range(slots_per_gpu)]
    return {s: jobs[i::len(slots)] for i, s in enumerate(slots) if jobs[i::len(slots)]}

L = {"catalina": 100, "ztf": 40}[DATASET]     # points per curve
CLASS_RANGE_K, DATA_SEED = 2.0, 0             # Catalina pre-processing, as in notebook 01

LOCAL_TEST = os.environ.get("ASTRA_LOCAL_TEST") == "1"     # only my offline test run sets this
RUN_STARTED = time.time()
WORK = Path(os.environ.get("ASTRA_WORK", "/kaggle/working"))
INPUT = Path(os.environ.get("ASTRA_INPUT", "/kaggle/input"))
TAG = f"{FAMILY}_{DATASET}"
# one runs folder per stage, so final runs start fresh (smoke outputs never mix with the real ones)
STAGE_RUNS = {s: WORK / (f"runs_{TAG}" + ("_final" if s == "final" else "") + ("_smoke" if SMOKE else ""))
              for s in ("search", "final")}
RUNS = STAGE_RUNS[STAGE]
RESULTS = WORK / (f"results_{DATASET}" + ("_smoke" if SMOKE else ""))   # one file per finished job (all notebooks alike)
FIGS, TABLES = WORK / f"figures_{TAG}", WORK / f"tables_{TAG}"
for d in (RUNS, RESULTS, FIGS, TABLES, WORK / "data", WORK / "astra"):
    d.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(WORK))
assert STAGE in ("search", "final") and (STAGE == "search" or FINAL_JOBS or SMOKE), \
    "STAGE = 'final' needs the search's winner: FINAL in astra/bench_grid.py, then a rebuilt notebook"
assert SEARCH_CONFIGS == "all" or set(SEARCH_CONFIGS) <= set(GRID_TAGS), f"unknown grid keys; choose from {GRID_TAGS}"
SEEDS = [0, 1] if SMOKE else sorted({int(j.rsplit("_s", 1)[1]) for j in JOBS_ALL})   # SMOKE: tiny search + final
print(f"{FAMILY} on {DATASET}: {STAGE} stage, session {SESSION} ({SESSION_HOURS} h), "
      f"{'SMOKE' if SMOKE else 'REAL RUN'} | {len(JOBS_ALL)} jobs: {JOBS_ALL}")
