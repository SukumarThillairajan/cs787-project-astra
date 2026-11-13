# ---- GPUs, and restoring the previous session's output ----
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
print("torch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert LOCAL_TEST or torch.cuda.device_count() >= 2, "Set Accelerator to GPU T4 x2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def restore_previous():
    """Copies the folders of an attached previous version of THIS notebook and stage (recognised by this
    stage's runs folder), so finished jobs are skipped and unfinished ones resume. Files already present are
    kept. Outputs of the other benchmark notebooks, and a search output attached to a final session, are not
    copied: the results section reads them where they are."""
    n = 0
    for runs in INPUT.glob(f"*/**/{RUNS.name}"):
        if not runs.is_dir() or WORK in runs.parents:
            continue
        for name in (RUNS.name, "data", f"results_{DATASET}", FIGS.name, TABLES.name):
            root = runs.parent / name
            for src in (root.rglob("*") if root.is_dir() else []):
                dest = WORK / name / src.relative_to(root)
                if src.is_file() and not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                    n += 1
    return n


n = restore_previous()
print(f"restored {n} file(s) from the previous session" if n else "fresh start (nothing to restore)")
for p in sorted(RUNS.glob("*/status.json")):
    print(f"  {p.parent.name:<14} {json.loads(p.read_text())}")
