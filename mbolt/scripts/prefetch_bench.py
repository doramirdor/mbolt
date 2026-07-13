"""The harvest benchmark: {orig, mbolt} x {prefetch on, off}, CPU mode,
24 GB mlocked squeeze. If explicit slice reads are what the layout needs,
prefetch-on should beat prefetch-off on both files, and mbolt+prefetch
should beat orig+prefetch (merged ranges collapse into fewer, larger reads).
"""

import os
import re
import subprocess
import sys
import time

import numpy as np

BIN = "/Users/dor/Documents/code/GPUopt/llama.cpp/build/bin/llama-cli"
PROMPT = "Explain the CAP theorem and give a concrete example of a CP and an AP system."
N_TOKENS = 128
N_REPS = 3

HOLDER_CODE = r"""
import ctypes, ctypes.util, sys, time
import numpy as np
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
gb = int(sys.argv[1])
hunks, locked = [], 0
rng = np.random.default_rng(1)
for i in range(gb):
    a = rng.integers(0, 2**63, size=(1 << 27,), dtype=np.int64)
    hunks.append(a)
    if libc.mlock(a.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(a.nbytes)) == 0:
        locked += 1
print(f"held {gb} GB, mlocked {locked} GB", flush=True)
while True:
    time.sleep(10)
"""


def vmstat(field: str) -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith(field):
            return int(line.split()[-1].rstrip("."))
    return -1


def evict():
    hunks = []
    try:
        for i in range(24):
            hunks.append(np.ones(1 << 27, np.float64))
            if i % 6 == 5 and vmstat("File-backed pages") * 16384 / 1e9 < 2.5:
                break
    except MemoryError:
        pass
    del hunks


def run_once(model: str, sidecar: str | None):
    env = dict(os.environ)
    env.pop("MBOLT_PREFETCH", None)
    if sidecar:
        env["MBOLT_PREFETCH"] = sidecar
    p0 = vmstat("Pageins")
    cmd = [BIN, "-m", model, "-p", PROMPT, "-n", str(N_TOKENS), "-st", "--temp", "0",
           "-ot", r".ffn_.*_exps.=CPU", "-c", "1024", "--no-warmup", "-ngl", "0", "-v"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=2400, env=env)
    text = out.stdout + out.stderr
    p1 = vmstat("Pageins")
    m = re.search(r"Generation:\s*([0-9.]+)\s*t/s", text)
    ts = float(m.group(1)) if m else -1
    pf = re.search(r"mbolt-prefetch:\s*(\d+) reads, ([0-9.]+) GB read, (\d+) slices already resident", text)
    pf_stats = f"pf: {pf.group(1)} reads / {pf.group(2)} GB / {pf.group(3)} resident" if pf else "pf: off"
    return ts, (p1 - p0) * 16384 / 1e9, pf_stats


def main():
    orig, opt, orig_pf, opt_pf = sys.argv[1:5]
    holder = subprocess.Popen([sys.executable, "-c", HOLDER_CODE, sys.argv[5] if len(sys.argv) > 5 else "24"],
                              stdout=subprocess.PIPE, text=True)
    print(holder.stdout.readline().strip(), flush=True)

    configs = [
        ("orig      ", orig, None),
        ("orig +pf  ", orig, orig_pf),
        ("mbolt     ", opt, None),
        ("mbolt+pf  ", opt, opt_pf),
    ]
    results = {name: [] for name, _, _ in configs}
    try:
        for rep in range(N_REPS):
            order = configs if rep % 2 == 0 else configs[::-1]
            for name, model, sidecar in order:
                evict()
                ts, gb_in, pf_stats = run_once(model, sidecar)
                results[name].append(ts)
                print(f"rep {rep} {name}: {ts:6.2f} tok/s | pageins {gb_in:6.2f} GB | {pf_stats}", flush=True)
    finally:
        holder.terminate()

    print()
    med = {}
    for name, vals in results.items():
        med[name] = np.median(vals)
        print(f"{name}: median {med[name]:.2f} tok/s, runs {sorted(round(v,2) for v in vals)}")
    print(f"\nprefetch effect on orig : {med['orig +pf  ']/med['orig      ']:.3f}x")
    print(f"prefetch effect on mbolt: {med['mbolt+pf  ']/med['mbolt     ']:.3f}x")
    print(f"layout win with explicit reads (mbolt+pf vs orig+pf): {med['mbolt+pf  ']/med['orig +pf  ']:.3f}x")
    print(f"full stack (mbolt+pf vs orig no-pf): {med['mbolt+pf  ']/med['orig      ']:.3f}x")


if __name__ == "__main__":
    main()
