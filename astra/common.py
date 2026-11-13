"""Small helpers shared by every notebook: seeds, paths, time guard, safe saving, logging."""
import json
import os
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

WORK = Path(os.environ.get("ASTRA_WORK", "/kaggle/working"))
INPUT = Path(os.environ.get("ASTRA_INPUT", "/kaggle/input"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    """Write to a temp file first, so a kill mid-write never leaves a broken file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def restore_rng(state, device):
    """Put back the random-number state saved in a resumable checkpoint. torch.load(map_location=GPU) moves
    the saved states to the GPU, but the generators only accept CPU tensors, so they are moved back first.
    Never fatal: without it a resumed run just continues with fresh random numbers. Returns True on success."""
    try:
        torch.set_rng_state(state["rng_cpu"].cpu())
        if state.get("rng_cuda") is not None and str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in state["rng_cuda"]])
        return True
    except Exception as e:
        print(f"could not restore the random-number state ({type(e).__name__}: {e}); continuing with a fresh one",
              flush=True)
        return False


class Deadline:
    """Session time guard: Kaggle kills a session at 12 h, we stop cleanly before that."""

    def __init__(self, deadline_unix):
        self.t = float(deadline_unix)

    def left(self):
        return self.t - time.time()

    def passed(self, reserve_s=0.0):
        return self.left() < reserve_s


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *msg):
        line = time.strftime("%H:%M:%S ") + " ".join(str(m) for m in msg)
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def find_input(name):
    """A file attached under /kaggle/input (e.g. a previous version's output), if any."""
    hits = sorted(INPUT.glob(f"**/{name}")) if INPUT.exists() else []
    return hits[0] if hits else None


def download(url, dest, tries=5):
    """Download once; reuse a copy attached under /kaggle/input if there is one."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    attached = find_input(dest.name)
    if attached is not None:
        return attached
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(tries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)
            return dest
        except Exception as e:  # network hiccup: wait and retry
            print(f"download failed ({e}), retry {i + 1}/{tries}")
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"could not download {url}")


def peak_gpu_mem_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9
