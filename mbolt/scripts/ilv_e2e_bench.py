"""E2E: stock vs chain+pipeline vs interleave, llama.cpp + MBOLT_PREFETCH,
CPU experts, 24 GB mlocked squeeze. Also fault-streaming (no-pf) controls -
interleave may help even there (per-expert region faults are sequential)."""

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

CONFIGS = [
    ("orig      ", f"{M}/Qwen3-Next-80B-A3B-Instruct-UD-IQ3_XXS.gguf", None),
    ("orig +pf  ", f"{M}/Qwen3-Next-80B-A3B-Instruct-UD-IQ3_XXS.gguf", f"{R}/orig.pf"),
    ("mbolt+pf  ", f"{M}/Qwen3-Next-80B-A3B-Instruct-UD-IQ3_XXS.mbolt.gguf", f"{R}/mbolt.pf"),
    ("ilv       ", f"{M}/Qwen3-Next-80B-A3B-Instruct-UD-IQ3_XXS.ilv.gguf", None),
    ("ilv  +pf  ", f"{M}/Qwen3-Next-80B-A3B-Instruct-UD-IQ3_XXS.ilv.gguf", f"{R}/ilv.pf"),
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


def run_once(model, sidecar):
    env = dict(os.environ)
    env.pop("MBOLT_PREFETCH", None)
    if sidecar:
        env["MBOLT_PREFETCH"] = sidecar
    cmd = [BIN, "-m", model, "-p", PROMPT, "-n", str(N_TOKENS), "-st", "--temp", "0",
           "-ot", r".ffn_.*_exps.=CPU", "-c", "1024", "--no-warmup", "-ngl", "0", "-v"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=2400, env=env)
    text = out.stdout + out.stderr
    m = re.search(r"Generation:\s*([0-9.]+)\s*t/s", text)
    pf = re.search(r"mbolt-prefetch:\s*(\d+) reads, ([0-9.]+) GB read", text)
    return (float(m.group(1)) if m else -1,
            f"pf: {pf.group(1)} reads / {pf.group(2)} GB" if pf else "pf: off")


def main():
    holder = subprocess.Popen([sys.executable, "-c", HOLDER_CODE, "24"],
                              stdout=subprocess.PIPE, text=True)
    print(holder.stdout.readline().strip(), flush=True)
    results = {name: [] for name, _, _ in CONFIGS}
    try:
        for rep in range(N_REPS):
            order = CONFIGS if rep % 2 == 0 else CONFIGS[::-1]
            for name, model, sidecar in order:
                evict()
                ts, pf = run_once(model, sidecar)
                results[name].append(ts)
                print(f"rep {rep} {name}: {ts:6.2f} tok/s | {pf}", flush=True)
    finally:
        holder.terminate()
    print()
    med = {n: np.median(v) for n, v in results.items()}
    for n, v in results.items():
        print(f"{n}: median {med[n]:.2f} tok/s, runs {sorted(round(x,2) for x in v)}")
    print(f"\nilv+pf vs orig      : {med['ilv  +pf  ']/med['orig      ']:.3f}x")
    print(f"ilv+pf vs orig+pf   : {med['ilv  +pf  ']/med['orig +pf  ']:.3f}x")
    print(f"ilv+pf vs mbolt+pf  : {med['ilv  +pf  ']/med['mbolt+pf  ']:.3f}x")
    print(f"ilv faults vs orig faults: {med['ilv       ']/med['orig      ']:.3f}x")


if __name__ == "__main__":
    main()
