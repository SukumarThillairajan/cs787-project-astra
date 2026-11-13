"""Which build ran? Compare the code cells of a notebook from Kaggle with a notebook built here.

    python build/nb_same_build.py "<notebook from Kaggle>.ipynb" notebooks/ASTRA_04_BENCH_VAE.ipynb

Prints, per differing code cell, the changed lines (built -> Kaggle). In the first code cell only the settings
(SMOKE, STAGE, SESSION, SESSION_HOURS, SLOTS_PER_GPU, SEARCH_CONFIGS, FINAL_SPLITS) may differ: a run changes them.
Ends with SAME BUILD or DIFFERENT BUILD. Reads the files as JSON only (nothing in them is run).
"""
import difflib
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
SETTINGS = re.compile(r"^(SMOKE|STAGE|SESSION|SESSION_HOURS|SLOTS_PER_GPU|SEARCH_CONFIGS|FINAL_SPLITS) = ")


def code_cells(path):
    nb = json.load(open(path, encoding="utf-8"))
    cells = ["".join(c["source"]).replace("\r", "") for c in nb["cells"] if c["cell_type"] == "code"]
    return [c for c in cells if c.strip()]          # an empty cell added on Kaggle is not a different build


kaggle, built = code_cells(sys.argv[1]), code_cells(sys.argv[2])
same = len(kaggle) == len(built)
print(f"code cells: Kaggle {len(kaggle)}, built {len(built)}")
for i, (b, k) in enumerate(zip(built, kaggle)):
    if b == k:
        continue
    diff = [line for line in difflib.unified_diff(b.splitlines(), k.splitlines(), lineterm="", n=0)
            if line[:1] in "+-" and line[:3] not in ("+++", "---")]
    settings_only = all(SETTINGS.match(line[1:]) for line in diff)
    same &= settings_only
    print(f"cell {i}: {'settings only' if settings_only else 'CODE DIFFERS'}")
    for line in diff[:12]:
        print("   ", line[:160])
print("SAME BUILD" if same else "DIFFERENT BUILD")
