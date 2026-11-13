"""Summarise the Kaggle log of an ASTRA notebook: key lines once, the result tables, the end state, and
where the time went. Works for finished, failed and cancelled versions.

    python build/kaggle_log.py "Logs/<file>.log"            # summary
    python build/kaggle_log.py "Logs/<file>.log" --ckpt     # + every checkpoint line of every GAN job, once
"""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
path, ckpt = sys.argv[1], "--ckpt" in sys.argv
rows = []
text = open(path, encoding="utf-8", errors="replace").read()
if text.lstrip().startswith("[{"):  # the JSON log that `kaggle kernels output` saves
    import json
    for e in json.loads(text):
        for s in e["data"].splitlines():
            rows.append((float(e["time"]), s))
else:  # the text log of the version page's Logs -> Download
    for line in text.splitlines():
        m = re.match(r"([0-9.]+)s (\d+) (.*)", line)
        if m:
            rows.append((float(m.group(1)), m.group(3)))
NOISE = ("Debugger warning", "frozen modules", "to python to disable", "Debugging will proceed", "Writing astra/",
         "NbConvertApp", "SyntaxWarning", "re.sub(")
rows = [(t, s) for t, s in rows if not any(n in s for n in NOISE)]
STATUS = re.compile(r"\[ *[0-9.]+ h\] GPU (\d) (running|exited ) \| ([^:]+): (.*)")
KEY = ("session ", "torch ", "restored ", "fresh start", "training file", "light curves of", " curves x ", "resume check",
       "self-tests", "canary", "every job is already done", "jobs to run", "workers finished", "it/s", "one GAN job",
       "GPU 0: [", "GPU 1: [", "Table 3: fitting", "output size", "DELIVERABLES", "MISSING", "unfinished", "!!",
       "Traceback", "Error", "failed", "{'state'")

print(f"== {path}\n   {len(rows)} lines, last at {rows[-1][0] / 3600:.2f} h ({rows[-1][0]:.0f} s)\n")
print("-- key lines")
for t, s in rows:
    if not STATUS.match(s) and any(k in s for k in KEY) and '"state"' not in s:
        print(f"{t:9.1f}s  {s[:200]}")

print("\n-- jobs (from the status lines: first seen, last seen, last line)")
jobs = {}
for t, s in rows:
    m = STATUS.match(s)
    if m:
        j = jobs.setdefault(m.group(3), {"gpu": m.group(1), "first": t})
        j.update(last=t, line=m.group(4), exited=m.group(2).strip() == "exited")
for name, j in jobs.items():
    print(f"GPU {j['gpu']} {name:<18} {j['first'] / 3600:5.2f} h -> {j['last'] / 3600:5.2f} h | {j['line'][:140]}")
if ckpt:
    seen = set()
    for t, s in rows:
        m = STATUS.match(s)
        if m and "checkpoint" in m.group(4) and (m.group(3), m.group(4)) not in seen:
            seen.add((m.group(3), m.group(4)))
            print(f"   {m.group(3)}: {m.group(4)[:150]}")

print("\n-- tables")
show = False
for t, s in rows:
    if s.startswith(("Table 5", "Table 3", "Welch", "Per-class", "latent codes against")):
        show = True
    elif s.startswith(("{", "/tmp", "/usr", "/kaggle")) or STATUS.match(s):
        show = False
    if show and s.strip():
        print(s[:230])

print("\n-- longest gaps between two log lines (where the run sat)")
gaps = sorted(((rows[i + 1][0] - rows[i][0], i) for i in range(len(rows) - 1)), reverse=True)[:4]
for g, i in sorted(gaps, key=lambda x: x[1]):
    print(f"{g / 60:7.1f} min after {rows[i][0]:9.1f}s  {rows[i][1][:110]}")
