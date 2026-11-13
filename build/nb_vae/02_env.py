# ---- GPUs, and restoring the previous session's output ----
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
print("torch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert LOCAL_TEST or torch.cuda.device_count() >= 2, "Set Accelerator to GPU T4 x2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def restore_previous():
    """Copies runs/, figures/ and tables/ of an attached previous version (finished jobs are skipped,
    unfinished ones resume from their last epoch). Files already present are kept."""
    n = 0
    for name in ("runs", "figures", "tables"):
        for root in INPUT.glob(f"*/**/{name}"):
            if not root.is_dir() or WORK in root.parents:
                continue
            for src in root.rglob("*"):
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
