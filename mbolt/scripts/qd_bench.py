"""Queue-depth sweep: does parallelizing the prefetcher's merged reads
(QD>1) lift streaming decode, and does layout still matter at depth?"""

import os
import re
import subprocess
import sys
import time

import numpy as np

BIN = "/Users/dor/Documents/code/GPUopt/llama.cpp/build/bin/llama-cli"
M = "/Users/dor/Documents/code/GPUopt/models"
R = "/Users/dor/Documents/code/GPUopt/results/qwen80"
PROMPT = "Explain the CAP theorem and give a concrete example of a CP and an AP system."
N_TOKENS = 128
N_REPS = 3

ORIG = f"{M}/Qwen3-Next-80B-A3B-Instruct-UD-IQ3_XXS.gguf"
ILV = f"{M}/Qwen3-Next-80B-A3B-Instruct-UD-IQ3_XXS.ilv.gguf"

CONFIGS = [
    ("orig pf qd1", ORIG, f"{R}/orig.pf", 1),
    ("orig pf qd4", ORIG, f"{R}/orig.pf", 4),
    ("ilv  pf qd1", ILV, f"{R}/ilv.pf", 1),
    ("ilv  pf qd4", ILV, f"{R}/ilv.pf", 4),
    ("ilv  pf qd8", ILV, f"{R}/ilv.pf", 8),
]

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


def vmstat_fb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("File-backed pages"):
            return int(line.split()[-1].rstrip(".")) * 16384 / 1e9
    return -1


def evict():
    hunks = []
    try:
        for i in range(24):
            hunks.append(np.ones(1 << 27, np.float64))
            if i % 6 == 5 and vmstat_fb() < 2.5:
                break
    except MemoryError:
        pass
    del hunks


def run_once(model, sidecar, threads):
    env = dict(os.environ, MBOLT_PREFETCH=sidecar, MBOLT_PREFETCH_THREADS=str(threads))
    cmd = [BIN, "-m", model, "-p", PROMPT, "-n", str(N_TOKENS), "-st", "--temp", "0",
           "-ot", r".ffn_.*_exps.=CPU", "-c", "1024", "--no-warmup", "-ngl", "0", "-v"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=2400, env=env)
    text = out.stdout + out.stderr
    m = re.search(r"Generation:\s*([0-9.]+)\s*t/s", text)
    pf = re.search(r"mbolt-prefetch:\s*(\d+) reads, ([0-9.]+) GB read", text)
    return (float(m.group(1)) if m else -1,
            f"{pf.group(1)} reads / {pf.group(2)} GB" if pf else "n/a")


def main():
    holder = subprocess.Popen([sys.executable, "-c", HOLDER_CODE, "24"],
                              stdout=subprocess.PIPE, text=True)
    print(holder.stdout.readline().strip(), flush=True)
    results = {name: [] for name, *_ in CONFIGS}
    try:
        for rep in range(N_REPS):
            order = CONFIGS if rep % 2 == 0 else CONFIGS[::-1]
            for name, model, sidecar, threads in order:
                evict()
                ts, pf = run_once(model, sidecar, threads)
                results[name].append(ts)
                print(f"rep {rep} {name}: {ts:6.2f} tok/s | pf {pf}", flush=True)
    finally:
        holder.terminate()
    print()
    med = {n: np.median(v) for n, v in results.items()}
    for n, v in results.items():
        print(f"{n}: median {med[n]:.2f} tok/s, runs {sorted(round(x,2) for x in v)}")
    print(f"\nQD4 vs QD1 (orig): {med['orig pf qd4']/med['orig pf qd1']:.3f}x")
    print(f"QD4 vs QD1 (ilv) : {med['ilv  pf qd4']/med['ilv  pf qd1']:.3f}x")
    print(f"QD8 vs QD4 (ilv) : {med['ilv  pf qd8']/med['ilv  pf qd4']:.3f}x")
    print(f"layout at QD4    : {med['ilv  pf qd4']/med['orig pf qd4']:.3f}x")


if __name__ == "__main__":
    main()
